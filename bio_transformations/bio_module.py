from __future__ import annotations

import logging
from typing import Callable
import math
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from typing import Callable, Optional

from bio_transformations.bio_config import BioConfig, DEFAULT_BIO_CONFIG, Distribution


class BioModule(nn.Module):
    """
    A module that provides bio-inspired modifications to standard PyTorch modules.

    This module implements various biologically-inspired learning mechanisms based on
    dendritic spine dynamics research. It enhances standard PyTorch modules with
    mechanisms that mimic:
    1. Synaptic diversity via fuzzy learning rates
    2. Structural plasticity via weight rejuvenation
    3. Synaptic stabilization via crystallization
    4. Volume-dependent plasticity rules

    When attached to a PyTorch module (like nn.Linear or nn.Conv2d), it enables
    these biological mechanisms during the training process.
    """

    # List of exposed functions that can be called on converted model layers
    # These functions are made available at the model level by BioConverter
    exposed_functions = (
        "rejuvenate_weights",
        "crystallize",
        "fuzzy_learning_rates",
        "volume_dependent_lr",
        "update_fuzzy_learning_rates",
        "volume_dependent_lr",
        "rejuvenate_weights_old"
    )

    def __init__(self, parent: Callable[[], nn.Module], config: BioConfig = DEFAULT_BIO_CONFIG) -> None:
        """
        Initializes the BioModule with a parent module and configuration.

        Args:
            parent: Callable returning the parent module to which the BioModule is attached.
                   This callable pattern allows for deferred access to the parent.
            config: BioConfig object containing the parameters for all bio-inspired mechanisms.
                   Defaults to the predefined DEFAULT_BIO_CONFIG if not specified.
        """
        super().__init__()
        self.get_parent = parent
        self.config = config

        # Ensure all parameters are valid before proceeding
        self._validate_parameters()

        # Initialize Dale's principle if enabled (applies neuron-specific constraints)
        if self.config.apply_dales_principle:
            self.apply(self.dalian_network_initialization)

        # Set up the parameters for fuzzy learning rates
        self._initialize_fuzzy_learning_rate_parameters()

    def _validate_parameters(self) -> None:
        """
        Validates the input parameters to ensure they meet required constraints.

        Raises:
            AssertionError: If any parameter value is invalid.
        """
        assert callable(self.get_parent), "parent must be a Callable returning an nn.Module instance"
        assert isinstance(self.get_parent(), nn.Module), "parent output must be an instance of nn.Module"
        assert self.config.fuzzy_learning_rate_factor_nu > 0, "fuzzy_learning_rate_factor_nu must be positive"
        assert 0 < self.config.dampening_factor < 1, "dampening_factor must be between 0 and 1"
        assert self.config.crystal_thresh > 0, "crystal_thresh must be positive"
        assert self.config.rejuvenation_parameter_dre > 0, "rejuvenation_parameter_dre must be positive"
        assert self.config.weight_splitting_Gamma >= 0, "weight_splitting_Gamma cannot be negative"
        assert callable(
            self.config.weight_splitting_activation_function), "weight_splitting_activation_function must be Callable"
        assert 0 < self.config.base_lr < 1, "base_lr must be between 0 and 1"
        assert self.config.stability_factor > 0, "stability_factor must be positive"
        assert self.config.lr_variability > 0, "lr_variability must be positive"

    def rejuvenate_weights_old(self) -> None:
        """
        Legacy implementation of weight rejuvenation.

        This function replaces weights below a certain threshold with random values,
        mimicking the biological process of synaptic pruning and regrowth.

        Note: This is kept for backward compatibility. Use rejuvenate_weights() instead.
        """

        with torch.no_grad():
            weight = self.get_parent().weight.data
            # Clamp weights to prevent extreme values
            weight = torch.clamp(weight, min=-1, max=1)
            mean, max_weight, std = weight.mean(), min(1.0, weight.abs().max()), weight.std()

            # Compute a threshold based on the rejuvenation parameter (dre)
            # Higher dre values result in less rejuvenation
            rejuvenation_threshold = torch.exp(
                torch.normal(mean, abs(max_weight / self.config.rejuvenation_parameter_dre) + 1e-13,
                             weight.shape)).to(weight.device).abs()

            # Transform parameters for log-normal distribution to maintain weight statistics
            std = torch.log(1 + (std ** 2) / (mean ** 2))  # Variance of log-normal = exp(sigma^2) - 1
            mean = torch.log(mean) - 0.5 * std  # Adjust mean to account for log transformation
            std = torch.sqrt(std)  # Convert back to standard deviation

            # Generate random weights from log-normal distribution
            random_weights = torch.exp(torch.normal(mean, std, size=weight.shape))

            # Create binary mask for weights below threshold
            mask = rejuvenation_threshold > weight.abs()

            # Replace weights below threshold with new random weights
            weight[mask] = random_weights[mask]
            logging.info(f"Rejuvenated {mask.sum().item()} weights")
            self.get_parent().weight.data = weight

    def rejuvenate_weights(self) -> None:
        """
        Rejuvenates the weights of the parent module.

        This function implements biologically-inspired structural plasticity by
        replacing weaker synapses with new ones. Key features:

        1. Smaller weights have higher probability of being replaced (like weak spines)
        2. Replaced weights are drawn from a log-normal distribution to match the
           overall weight statistics (mimicking new spine formation)
        3. The rejuvenation parameter (dre) controls how aggressive the rejuvenation is:
           - Higher values preserve more weights (less turnover)
           - Lower values cause more weights to be replaced (higher turnover)

        This mimics the biological process where weak or unused synapses are pruned
        while new synapses form, allowing the network to explore new connectivity patterns.
        """

        with torch.no_grad():
            weight = self.get_parent().weight.data

            # Handle NaN values in the weight matrix by replacing them with zeros
            weight = torch.nan_to_num(weight, nan=0.0)

            # Clamp weights to prevent extreme values
            weight = torch.clamp(weight, min=-1, max=1)

            # Calculate weight statistics for new weight generation
            mean = weight.mean().item()
            std = weight.std().item()

            # Avoid numerical issues with very small or negative values
            if mean <= 0 or torch.isnan(torch.tensor(mean)):
                mean = 1e-8
            if std <= 0 or torch.isnan(torch.tensor(std)):
                std = 1e-8

            mean_t = torch.tensor(mean, device=weight.device)
            std_t = torch.tensor(std, device=weight.device)

            # Compute log-normal parameters to maintain weight distribution statistics
            # These formulas convert from normal distribution to log-normal distribution
            log_variance = torch.log(1 + (std_t ** 2) / (mean_t ** 2))
            log_mean = torch.log(mean_t) - 0.5 * log_variance
            log_std = torch.sqrt(log_variance)

            # Get rejuvenation parameter (dre) - controls probability of weight replacement
            dre = self.config.rejuvenation_parameter_dre

            # Calculate rejuvenation probability inversely proportional to weight magnitude
            # Smaller weights have higher probability of being replaced
            # The exponential function ensures probabilities are between 0 and 1
            probability = torch.exp(- (dre * weight.abs() / std))

            # Handle edge cases for numerical stability
            probability = torch.nan_to_num(probability, nan=0.0, posinf=0.0, neginf=0.0)
            probability = torch.clamp(probability, 0.0, 1.0)

            # Create a mask for NaN values in the original weight
            nan_mask = torch.isnan(self.get_parent().weight.data)

            # Create the rejuvenation mask by sampling from Bernoulli distribution
            # Each weight has probability of being 1 (rejuvenated) based on calculated probability
            # Also include any NaN positions in the mask for definite replacement
            rejuvenation_mask = torch.bernoulli(probability).bool() | nan_mask

            # Generate new random weights from log-normal distribution
            # Using half the std for more stability in the new weights
            random_weights = torch.exp(torch.normal(log_mean, log_std * 0.5, size=weight.shape, device=weight.device))
            random_weights = torch.clamp(random_weights, min=-1, max=1)

            # Apply the mask: replace selected weights with new random values
            weight[rejuvenation_mask] = random_weights[rejuvenation_mask]

            rejuvenated_count = rejuvenation_mask.sum().item()
            logging.info(f"Rejuvenated {rejuvenated_count} weights")

            self.get_parent().weight.data = weight

    def crystallize(self) -> None:
        """
        Crystallizes (stabilizes) the weights of the parent module by adjusting gradient scaling.

        This process mimics the biological phenomenon of synaptic stabilization,
        where frequently used synapses become less plastic over time. Key features:

        1. Identifies weights with small gradients relative to their magnitude,
           which indicates they are already well-optimized or important
        2. Also identifies weights with unusually large gradients, which may
           indicate important learning signals
        3. For these weights, reduces their fuzzy learning rate parameter,
           effectively making them more stable and resistant to change
        4. The dampening_factor controls how much stability is increased each time

        This helps the network retain important learned patterns while still
        allowing exploration of new patterns through other weights.

        Raises:
            RuntimeError: If weights don't require gradients or no gradients are found.
        """
        if not self.get_parent().weight.requires_grad:
            raise RuntimeError("Weights do not require gradients")

        if self.get_parent().weight.grad is None:
            raise RuntimeError("No gradients found for the weights")

        with torch.no_grad():
            weight = self.get_parent().weight.data.abs()
            grad = self.get_parent().weight.grad.abs()
            mean_weight = weight.mean()

            # Identify weights to crystallize based on two criteria:
            # 1. Weights with small gradients relative to their size (grad/weight < threshold)
            # 2. Weights with unusually large gradients (grad > mean_weight)
            mask = ((grad / weight) < self.config.crystal_thresh) | (grad > mean_weight)

            # Apply the dampening factor to reduce learning rates for crystallized weights
            self.fuzzy_learning_rate_parameters[mask] *= self.config.dampening_factor

    def volume_dependent_lr(self) -> None:
        """
        Applies a volume-dependent learning rate to the weights of the parent module.

        This method implements a biologically-inspired learning rate adjustment based on
        observations of dendritic spine dynamics. It reflects the following key findings:

        1. Larger weights (analogous to larger spines) are more stable and less plastic.
           The stability increases exponentially with weight size.
        2. Smaller weights (analogous to smaller spines) are more dynamic and plastic.
        3. There is significant variability in plasticity among weights of similar sizes.
        4. The relationship between weight size and plasticity is continuous, not discrete.

        The implementation:
        - Normalizes weights to [0,1] range to represent relative "volumes"
        - Calculates a base learning rate that decreases exponentially with volume
        - Adds variability by sampling from a normal distribution
        - Applies these factors to weight gradients

        Parameters controlling this behavior:
        - base_lr: Base learning rate, higher values increase plasticity overall
        - stability_factor: Controls how quickly stability increases with weight size
        - lr_variability: Controls the amount of variability in learning rates

        Raises:
            RuntimeError: If no gradients are found for the weights.
        """
        if self.get_parent().weight.grad is None:
            raise RuntimeError("No gradients found for the weights")

        with torch.no_grad():
            # Get absolute weight values (analogous to spine volume)
            weight_abs = self.get_parent().weight.data.abs()

            # Normalize weights to [0, 1] range using sigmoid-like function
            # This represents relative "volume" of each weight/synapse
            normalized_weights = weight_abs / (weight_abs + 1)

            # Calculate mean learning rate factor based on normalized weights
            # Larger weights get exponentially smaller learning rates
            # stability_factor controls how quickly learning rates decrease with size
            lr_mean = self.config.base_lr * torch.exp(-self.config.stability_factor * normalized_weights)

            # Calculate standard deviation for learning rate variability
            # Variability is proportional to mean learning rate
            # lr_variability controls the amount of variability
            lr_std = self.config.lr_variability * lr_mean

            # Sample learning rates from normal distribution for each weight
            # Take absolute value to ensure all learning rates are positive
            lr_factors = torch.abs(torch.normal(lr_mean, lr_std))

            # Apply the learning rate factors to the gradients
            # This effectively gives each weight its own learning rate
            self.get_parent().weight.grad *= lr_factors

    def _initialize_fuzzy_learning_rate_parameters(self) -> None:
        """
        Initializes the fuzzy learning rate parameters based on the selected distribution.

        This method creates parameter tensors that will be used to scale gradients during
        training, introducing biologically-inspired variability in learning rates.
        Different distribution strategies are available to model various biological
        phenomena:

        - BASELINE: No variability (all parameters = 1.0)
        - UNIFORM: Uniform distribution around 1.0
        - NORMAL: Normal distribution centered at 1.0
        - LOGNORMAL: Log-normal with mean 1.0 (skewed, only positive values)
        - GAMMA: Gamma distribution (positive, skewed)
        - BETA: Beta distribution scaled to [1-nu, 1+nu]
        - LAYER_ADAPTIVE: Layer-dependent variability (decreases with depth)
        - WEIGHT_ADAPTIVE: Weight-dependent scaling (smaller weights get more variability)
        - TEMPORAL: Evolves over time
        - ACTIVITY: Based on neuron activation patterns

        The resulting parameters are stored as non-trainable parameters in the module.
        They are used by fuzzy_learning_rates() to scale gradients during training.
        """
        shape = self.get_parent().weight.data.shape
        nu = self.config.fuzzy_learning_rate_factor_nu
        dist = self.config.fuzzy_lr_distribution

        # Initialize parameters based on selected distribution
        if dist == Distribution.BASELINE:
            # Simplest case: all parameters are exactly 1.0 (no variability)
            params = torch.ones_like(self.get_parent().weight.data)

        elif dist == Distribution.UNIFORM:
            # Uniform distribution in range [1-nu, 1+nu]
            # 2*torch.rand_like(...) gives values in [0, 2]
            # Multiply by nu to get [0, 2*nu]
            # Subtract nu to get [-nu, +nu]
            # Add 1 to center at 1.0, giving [1-nu, 1+nu]
            params = 1. + (2. * torch.rand_like(self.get_parent().weight.data) * nu - nu)

        elif dist == Distribution.NORMAL:
            # Normal distribution centered at 1.0 with standard deviation nu
            params = 1. + torch.normal(0, nu, size=shape, device=self.get_parent().weight.device)

        elif dist == Distribution.LOGNORMAL:
            # Log-normal with mean=1.0
            # For log-normal distribution, if X ~ N(μ, σ²) then Y = e^X ~ LogNormal(μ, σ²)
            # The mean of Y is e^(μ + σ²/2)
            # To get E[Y] = 1, we need μ = -σ²/2
            mu = -0.5 * (nu ** 2)  # This ensures the mean of e^N(mu,sigma) is 1.0
            params = torch.exp(torch.normal(mu, nu, size=shape, device=self.get_parent().weight.device))

        elif dist == Distribution.GAMMA:
            # Gamma distribution
            # For Gamma(k, θ), mean = k*θ and variance = k*θ²
            # Using concentration = rate = 1/nu gives a distribution with mean 1/nu * 1/nu = 1
            concentration = 1 / nu
            rate = 1 / nu
            gamma_dist = torch.distributions.Gamma(concentration=concentration, rate=rate)
            params = 1 + gamma_dist.sample(shape).to(self.get_parent().weight.device)

        elif dist == Distribution.BETA:
            # Beta distribution scaled to [1-nu, 1+nu]
            # Beta(α, β) with α = β gives a symmetric distribution centered at 0.5
            # We scale this to [1-nu, 1+nu] by: 1 + (2*beta_sample - 1)*nu
            alpha = beta = 1 / nu
            beta_dist = torch.distributions.Beta(alpha, beta)
            beta_samples = beta_dist.sample(shape).to(self.get_parent().weight.device)
            params = 1 + (2 * beta_samples - 1) * nu

        elif dist == Distribution.LAYER_ADAPTIVE:
            # Layer-adaptive sampling - deeper layers get less variability
            # This mimics the biological observation that deeper cortical layers
            # show less synaptic plasticity
            if self.config.fuzzy_lr_layer_index >= 0 and self.config.fuzzy_lr_total_layers > 0:
                # Calculate a layer factor that decreases with depth
                # First layer (index 0) has factor 1.0
                # Last layer (index total_layers-1) has factor 0.5
                layer_factor = 1 - 0.5 * self.config.fuzzy_lr_layer_index / self.config.fuzzy_lr_total_layers
                adaptive_nu = nu * layer_factor
                params = 1. + (2. * torch.rand_like(self.get_parent().weight.data) * adaptive_nu - adaptive_nu)
            else:
                logging.warning(
                    "Layer-adaptive requested but layer index or total not set. Using uniform distribution.")
                params = 1. + (2. * torch.rand_like(self.get_parent().weight.data) * nu - nu)

        elif dist == Distribution.WEIGHT_ADAPTIVE:
            # Initialize with ones first, will be updated in first training step
            # The actual weight-dependent initialization happens in _update_weight_adaptive_rates
            params = torch.ones_like(self.get_parent().weight.data)
            self.weight_adaptive_initialized = False

        elif dist == Distribution.TEMPORAL:
            # Start with uniform distribution, will evolve over time
            # The parameters will change during training based on update_fuzzy_learning_rates
            params = 1. + (2. * torch.rand_like(self.get_parent().weight.data) * nu - nu)
            self.update_step = 0

        elif dist == Distribution.ACTIVITY:
            # Start with ones, will be updated based on activation patterns
            # Updates occur in update_fuzzy_learning_rates based on input activity
            params = torch.ones_like(self.get_parent().weight.data)

            # Determine output dimension size for activation counting
            if isinstance(self.get_parent(), nn.Linear):
                output_size = self.get_parent().out_features
            elif isinstance(self.get_parent(), nn.Conv2d):
                output_size = self.get_parent().out_channels
            else:
                output_size = shape[0] if len(shape) > 0 else 0

            # Create a buffer to track activation counts for each output neuron
            self.register_buffer('activation_count', torch.zeros(output_size, device=self.get_parent().weight.device))
            self.activity_initialized = False

        else:
            logging.warning(f"Unknown distribution: {dist}. Using uniform distribution as fallback.")
            params = 1. + (2. * torch.rand_like(self.get_parent().weight.data) * nu - nu)

        # Clamp values to specified min/max range for numerical stability
        params = torch.clamp(params, min=self.config.fuzzy_lr_min, max=self.config.fuzzy_lr_max)

        # Create parameter tensor (not trainable)
        self.fuzzy_learning_rate_parameters = nn.Parameter(params, requires_grad=False)

    def update_fuzzy_learning_rates(self, x: Optional[torch.Tensor] = None) -> None:
        """
        Updates fuzzy learning rates based on the chosen distribution strategy.

        This method refreshes the fuzzy learning rate parameters according to the
        selected distribution strategy. Some strategies require updates during
        training (TEMPORAL, ACTIVITY), while others are static or only need
        initialization (BASELINE, UNIFORM, etc.).

        Args:
            x: Optional input tensor, needed for activity-based updates.
               This should be the input to the layer for accurate activity tracking.
        """
        if not self.config.fuzzy_lr_dynamic:
            # Only update for weight-adaptive during initialization
            if self.config.fuzzy_lr_distribution == Distribution.WEIGHT_ADAPTIVE and not getattr(self,
                                                                                                 'weight_adaptive_initialized',
                                                                                                 True):
                self._update_weight_adaptive_rates()
                self.weight_adaptive_initialized = True
            return

        dist = self.config.fuzzy_lr_distribution

        if dist == Distribution.TEMPORAL:
            # Temporal strategy - parameters evolve over time
            self._update_temporal_rates()

        elif dist == Distribution.ACTIVITY and x is not None:
            # Activity-dependent strategy - parameters depend on neuron activations
            self._update_activity_dependent_rates(x)

    def _update_temporal_rates(self) -> None:
        """
        Updates fuzzy learning rates for temporal evolution strategy.

        This implements a time-dependent evolution of learning rates, where:
        1. Existing rate factors gradually decay toward 1.0 (neutral value)
        2. Periodically, small random variations are injected to maintain diversity

        This mimics the biological phenomenon where synaptic plasticity has both
        stable trends and stochastic fluctuations over time.
        """
        self.update_step += 1

        # Only update periodically based on the configured frequency
        if self.update_step % self.config.fuzzy_lr_update_freq != 0:
            return

        with torch.no_grad():
            nu = self.config.fuzzy_learning_rate_factor_nu
            decay = self.config.fuzzy_lr_decay

            # Decay existing rates slightly toward 1.0
            # This pushes extreme values back toward the neutral value
            self.fuzzy_learning_rate_parameters.data = 1.0 + (self.fuzzy_learning_rate_parameters.data - 1.0) * decay

            # Add new randomness occasionally (every 10th update)
            # This maintains diversity in the learning rates
            if self.update_step % (self.config.fuzzy_lr_update_freq * 10) == 0:
                noise = 2 * torch.rand_like(self.fuzzy_learning_rate_parameters) * nu - nu
                self.fuzzy_learning_rate_parameters.data += noise * 0.1

            # Re-clamp to bounds to maintain stability
            self.fuzzy_learning_rate_parameters.data = torch.clamp(
                self.fuzzy_learning_rate_parameters.data,
                min=self.config.fuzzy_lr_min,
                max=self.config.fuzzy_lr_max
            )

    def _update_weight_adaptive_rates(self) -> None:
        """
        Updates fuzzy learning rates based on weight magnitudes.

        This implements weight-dependent learning rates where:
        1. Smaller weights get higher variability (more plasticity)
        2. Larger weights get lower variability (more stability)

        This mimics the biological observation that smaller synapses are more
        dynamic while larger synapses tend to be more stable.
        """
        with torch.no_grad():
            # Get weight magnitudes (absolute values)
            weight_magnitudes = self.get_parent().weight.data.abs()

            # Normalize by mean weight to get relative sizes
            mean_weight = weight_magnitudes.mean() + 1e-8  # Add small epsilon to avoid division by zero
            normalized_weights = weight_magnitudes / mean_weight

            # Calculate adaptive variability factor based on normalized weights
            # Formula: 2 - normalized_weights, clamped to range [0.5, 1.5]
            # Effect: Weights at mean magnitude (normalized=1) get variability nu
            #         Weights at 2x mean get variability 0.5*nu (more stable)
            #         Weights at 0.5x mean get variability 1.5*nu (more plastic)
            nu = self.config.fuzzy_learning_rate_factor_nu
            adaptive_nu = nu * torch.clamp(2 - normalized_weights, 0.5, 1.5)

            # Generate new rates from uniform distribution with adaptive variability
            new_rates = 1. + (2. * torch.rand_like(self.get_parent().weight.data) * adaptive_nu - adaptive_nu)

            # Clamp values to configured bounds
            self.fuzzy_learning_rate_parameters.data = torch.clamp(
                new_rates,
                min=self.config.fuzzy_lr_min,
                max=self.config.fuzzy_lr_max
            )

    def _update_activity_dependent_rates(self, x: torch.Tensor) -> None:
        """
        Updates fuzzy learning rates based on input activation patterns.

        This implements activity-dependent plasticity where:
        1. More frequently active neurons get more stable learning rates
        2. Less frequently active neurons get more variable learning rates

        This mimics the biological phenomenon where highly active neural pathways
        develop more consistent synaptic strengths.

        Args:
            x: Input tensor to the layer, used to determine activation patterns
        """
        with torch.no_grad():
            # Compute output without using the parent module directly to avoid recursion
            if isinstance(self.get_parent(), nn.Linear):
                # Manually compute the linear transformation
                weight = self.get_parent().weight
                bias = self.get_parent().bias if hasattr(self.get_parent(), 'bias') else None

                # output = x @ weight.T + bias
                output = torch.matmul(x, weight.t())
                if bias is not None:
                    output = output + bias

            elif isinstance(self.get_parent(), nn.Conv2d):
                # For Conv2d, we'll use functional interface but NOT call the module
                weight = self.get_parent().weight
                bias = self.get_parent().bias if hasattr(self.get_parent(), 'bias') else None
                stride = self.get_parent().stride
                padding = self.get_parent().padding
                dilation = self.get_parent().dilation
                groups = self.get_parent().groups

                output = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
            else:
                # Unsupported layer type
                return

            # Track which outputs are active (using a small threshold to detect activation)
            output_active = (output.abs() > 0.01).float()

            # Sum activations across batch dimension (and spatial dimensions for conv)
            # This gives a count of how often each output neuron was active
            if len(output_active.shape) > 2:  # For conv layers
                output_active = output_active.sum(dim=(0, 2, 3))
            else:
                output_active = output_active.sum(dim=0)

            # Update activation counts with current batch
            self.activation_count += output_active

            # Only update learning rates occasionally to reduce computation (10% chance)
            if not self.activity_initialized or random.random() < 0.1:
                nu = self.config.fuzzy_learning_rate_factor_nu

                # Normalize activation counts relative to the mean
                # This identifies which neurons are more/less active than average
                normalized_count = self.activation_count / (self.activation_count.mean() + 1e-8)

                # Apply learning rate adjustments based on output neuron activity
                for i in range(len(normalized_count)):
                    # Calculate activity factor: more active neurons have higher factors
                    # The 0.1 scaling dampens the effect for stability
                    activity_factor = 1 + (normalized_count[i] - 1) * 0.1
                    activity_factor = torch.clamp(activity_factor, 0.8, 1.2)

                    # For outputs with more activity, use lower variability (more stability)
                    # For less active outputs, use higher variability (more exploration)
                    local_nu = nu * (2 - activity_factor)

                    # Generate new random rates with activity-dependent variability
                    self.fuzzy_learning_rate_parameters.data[i, :] = 1 + (
                            torch.rand_like(self.fuzzy_learning_rate_parameters.data[i, :]) * 2 - 1
                    ) * local_nu

                # Clamp values to bounds for stability
                self.fuzzy_learning_rate_parameters.data = torch.clamp(
                    self.fuzzy_learning_rate_parameters.data,
                    min=self.config.fuzzy_lr_min,
                    max=self.config.fuzzy_lr_max
                )

                self.activity_initialized = True

    def fuzzy_learning_rates(self) -> None:
        """
        Scales the gradients of the parent module using fuzzy learning rates.

        This method introduces controlled variability into the learning process by:
        1. Multiplying each weight's gradient by its corresponding fuzzy learning rate
        2. These rates vary based on the distribution strategy chosen in the config

        This mimics the biological phenomenon where synapses have diverse plasticity
        characteristics, even between synapses of the same neuron pair.

        The fuzzy learning rate parameters are initialized by _initialize_fuzzy_learning_rate_parameters
        and can be dynamically updated by update_fuzzy_learning_rates during training.

        Raises:
            RuntimeError: If no gradients are found for the weights.
        """
        if self.get_parent().weight.grad is None:
            raise RuntimeError("No gradients found for the weights")

        # Apply the fuzzy learning rates by direct multiplication with gradients
        self.get_parent().weight.grad *= self.fuzzy_learning_rate_parameters


    def l1_reg(self) -> torch.Tensor:
        """
        Computes the L1 regularization of the module's parameters.

        L1 regularization encourages sparsity in the weights, which can mimic
        the biological principle of energy efficiency in neural systems.

        Returns:
            The L1 regularization value (sum of absolute values of all parameters).
        """
        with torch.no_grad():
            # Concatenate all parameters into a single vector
            all_params = torch.cat([x.view(-1) for x in self.parameters()])
            # Compute L1 norm (sum of absolute values)
            l1_regularization = torch.norm(all_params, 1)
        return l1_regularization

    @staticmethod
    def dalian_network_initialization(module: nn.Module) -> None:
        """
        Initializes the network weights according to Dale's principle.

        Dale's principle is a biological constraint stating that neurons release the same
        neurotransmitters at all of their synapses. In artificial networks, this is
        implemented by enforcing consistent sign for all outgoing weights from each neuron.

        The implementation:
        1. Takes the absolute values of all weights (making them positive)
        2. Assigns a random sign (+1 or -1) to each output neuron
        3. Applies these signs to the weights, ensuring each neuron has consistent output

        Args:
            module: The module to initialize. Must be a BioModule with a valid parent.

        Raises:
            AttributeError: If module is not a BioModule or parent is not a supported layer type.
        """
        if not isinstance(module, BioModule) or not isinstance(module.get_parent(), (nn.Linear, nn.Conv2d)):
            # Using the exact error message string expected by the test
            raise AttributeError(f"Can not use dalians network initialization on {type(module)}")

        weights = module.get_parent().weight.data

        # Take absolute values of weights (removing any existing sign information)
        weights = torch.abs(weights)

        # Determine the shape required for the sign tensor
        # For Conv2d, need 4D tensor [out_channels, in_channels, 1, 1]
        # For Linear, need 2D tensor [out_features, 1]
        shape = [weights.size(0), weights.size(1), 1, 1] if weights.ndim > 2 else [weights.size(0), 1]

        # Create a sign tensor with random +1/-1 values for each output neuron
        # This is crucial: each OUTPUT neuron gets ONE sign, applied to ALL its input weights
        module.sign = nn.Parameter(((torch.randint(0, 2, shape, dtype=torch.float) * 2) - 1), requires_grad=False).to(
            weights.device)

        # Apply the signs to the weights
        module.get_parent().weight.data = weights * module.sign

    def enforce_dales_principle(self) -> None:
        """
        Enforces Dale's principle on the weights of the parent module.

        This ensures that all outgoing weights from a neuron have the same sign,
        consistent with the biological principle that neurons release the same
        neurotransmitters at all of their synapses.

        In practice, this means:
        1. For excitatory neurons (positive sign), all outgoing weights are forced to be positive
        2. For inhibitory neurons (negative sign), all outgoing weights are forced to be negative

        This is achieved using a sign tensor created during dalian_network_initialization and
        applying ReLU with appropriate sign to force consistency.

        Raises:
            AttributeError: If apply_dales_principle is False or sign attribute is missing.
        """
        if not self.config.apply_dales_principle:
            # Using the exact error message string expected by the test
            raise AttributeError(f"Can not enforce dales principle without apply_dales_principle set True.")

        if not hasattr(self, 'sign'):
            raise AttributeError("sign attribute not found. Make sure dalian_network_initialization has been applied.")

        # Skip enforcing Dale's principle on specific modules marked with last_module_token
        # This typically applies to output layers where enforcing the principle might be undesirable
        if not hasattr(self.get_parent(), 'last_module_token'):
            # The implementation works by:
            # 1. Multiplying weights by the sign to make all values positive
            # 2. Applying ReLU to zero out any negative values
            # 3. Multiplying by sign again to restore the correct sign
            # This ensures all weights from a neuron have the same sign (all + or all -)
            self.get_parent().weight.data = torch.nn.functional.relu(
                self.get_parent().weight.data * self.sign) * self.sign