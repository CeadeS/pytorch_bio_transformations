from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn

from bio_transformations.bio_config import BioConfig, DEFAULT_BIO_CONFIG


class BioModule(nn.Module):
    """
    A module that provides bio-inspired modifications to standard PyTorch modules.

    This module implements various biologically-inspired learning mechanisms,
    including volume-dependent learning rates, fuzzy learning rates, and weight crystallization.
    """

    # List of exposed functions that can be called on converted model layers
    exposed_functions = ("rejuvenate_weights", "crystallize", "fuzzy_learning_rates", "volume_dependent_lr")

    def __init__(self, parent: Callable[[], nn.Module], config: BioConfig = DEFAULT_BIO_CONFIG) -> None:
        """
        Initializes the BioModule.

        Args:
            parent: Callable returning the parent module to which the BioModule is attached.
            config: BioConfig object containing the parameters.
        """
        super().__init__()
        self.get_parent = parent
        self.config = config

        self._validate_parameters()

        if self.config.apply_dales_principle:
            self.apply(self.dalian_network_initialization)

        self._initialize_fuzzy_learning_rate_parameters()

    def _validate_parameters(self) -> None:
        """Validates the input parameters."""
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

    def _initialize_fuzzy_learning_rate_parameters(self) -> None:
        """Initializes the fuzzy learning rate parameters."""
        self.fuzzy_learning_rate_parameters = nn.Parameter(1. + (2. * torch.rand_like(self.get_parent().weight.data,
                                                                                      requires_grad=False) * self.config.fuzzy_learning_rate_factor_nu - self.config.fuzzy_learning_rate_factor_nu))

    def rejuvenate_weights(self) -> None:
        """
        Rejuvenates the weights of the parent module.

        This function replaces weights below a certain threshold with random values,
        mimicking the biological process of synaptic pruning and regrowth.
        """
        if not self.get_parent().weight.requires_grad:
            raise RuntimeError("Weights do not require gradients")

        with torch.no_grad():
            weight = self.get_parent().weight.data
            mean, max_weight = 0.0, weight.abs().max()
            try:
                # Compute the rejuvenation threshold using a normal distribution
                rejuvenation_threshold = torch.normal(mean, max_weight / self.config.rejuvenation_parameter_dre + 1e-13,
                                                      weight.shape).to(weight.device).abs()
            except RuntimeError as e:
                raise RuntimeError("Error computing rejuvenation threshold.") from e

            # Generate random weights for rejuvenation
            random_weights = (torch.rand_like(weight) * 2 - 1) * max_weight
            mask = rejuvenation_threshold > weight.abs()
            weight[mask] = random_weights[mask]
            logging.info(f"Rejuvenated {mask.sum().item()} weights")
            self.get_parent().weight.data = weight

    def crystallize(self) -> None:
        """
        Crystallizes the weights of the parent module by adjusting gradient scaling.

        This process mimics the biological phenomenon of synaptic stabilization,
        where frequently used synapses become less plastic over time.
        """
        if not self.get_parent().weight.requires_grad:
            raise RuntimeError("Weights do not require gradients")

        with torch.no_grad():
            weight = self.get_parent().weight.data.abs()
            grad = self.get_parent().weight.grad.abs()
            mean_weight = weight.mean()
            mask = ((grad / weight) < self.config.crystal_thresh) | (grad > mean_weight)
            self.fuzzy_learning_rate_parameters[mask] *= self.config.dampening_factor

    def volume_dependent_lr(self) -> None:
        """
        Applies a volume-dependent learning rate to the weights of the parent module.

        This method implements a biologically-inspired learning rate adjustment based on
        observations of dendritic spine dynamics. It reflects the following key findings:
        1. Larger weights (analogous to larger spines) are more stable and less plastic.
        2. Smaller weights (analogous to smaller spines) are more dynamic and plastic.
        3. There is significant variability in plasticity among weights of similar sizes.
        4. The relationship between weight size and plasticity is continuous, not discrete.
        """
        if self.get_parent().weight.grad is None:
            raise RuntimeError("No gradients found for the weights")

        with torch.no_grad():
            weight_abs = self.get_parent().weight.data.abs()

            # Normalize weights to [0, 1] range, analogous to spine volume
            normalized_weights = weight_abs / (weight_abs + 1)

            # Calculate mean learning rate factor
            lr_mean = self.config.base_lr * torch.exp(-self.config.stability_factor * normalized_weights)

            # Calculate standard deviation for learning rate
            lr_std = self.config.lr_variability * lr_mean

            # Sample learning rates from normal distribution
            lr_factors = torch.abs(torch.normal(lr_mean, lr_std))

            # Apply the learning rate factors to the gradients
            self.get_parent().weight.grad *= lr_factors

    def fuzzy_learning_rates(self) -> None:
        """
        Scales the gradients of the parent module with random values.

        This method introduces stochasticity into the learning process,
        mimicking the variability observed in biological synaptic plasticity.
        """
        if self.get_parent().weight.grad is None:
            raise RuntimeError("No gradients found for the weights")

        self.get_parent().weight.grad *= self.fuzzy_learning_rate_parameters

    def l1_reg(self) -> torch.Tensor:
        """
        Computes the L1 regularization of the module's parameters.

        Returns:
            The L1 regularization value.
        """
        with torch.no_grad():
            all_params = torch.cat([x.view(-1) for x in self.parameters()])
            l1_regularization = torch.norm(all_params, 1)
        return l1_regularization

    @staticmethod
    def dalian_network_initialization(module: nn.Module) -> None:
        """
        Initializes the network weights according to Dale's principle.

        Dale's principle states that neurons release the same neurotransmitters
        at all of their synapses, which is reflected here by enforcing consistent
        sign for all outgoing weights from each neuron.

        Args:
            module: The module to initialize.
        """
        if not isinstance(module, BioModule) or not isinstance(module.get_parent(), (nn.Linear, nn.Conv2d)):
            raise AttributeError(f"Can not use dalians network initialization on {type(module)}")

        weights = module.get_parent().weight.data
        weights = torch.abs(weights)
        shape = [weights.size(0), weights.size(1), 1, 1] if weights.ndim > 2 else [weights.size(0), 1]
        module.sign = nn.Parameter(((torch.randint(0, 2, shape, dtype=torch.float) * 2) - 1), requires_grad=False).to(
            weights.device)
        module.get_parent().weight.data = weights * module.sign

    def enforce_dales_principle(self) -> None:
        """
        Enforces Dale's principle on the weights of the parent module.

        This ensures that all outgoing weights from a neuron have the same sign,
        consistent with the biological principle that neurons release the same
        neurotransmitters at all of their synapses.
        """
        if not self.config.apply_dales_principle:
            raise AttributeError(f"Can not enforce dales principle without apply_dales_principle set True.")
        if not hasattr(self, 'sign'):
            raise AttributeError("sign attribute not found. Make sure dalian_network_initialization has been applied.")
        self.get_parent().weight.data = torch.nn.functional.relu(self.get_parent().weight.data * self.sign) * self.sign
