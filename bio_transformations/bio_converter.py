from __future__ import annotations

import functools
from typing import Callable, Any, Type, Union

import torch
import torch.nn as nn

from bio_transformations.bio_config import BioConfig, DEFAULT_BIO_CONFIG, Distribution
from bio_transformations.bio_module import BioModule


class BioConverter:
    """
    A utility class that converts standard PyTorch modules to use bio-inspired learning mechanisms.

    BioConverter attaches BioModule instances to PyTorch modules (nn.Linear, nn.Conv2d),
    enabling biologically-inspired modifications like:

    1. Diverse synaptic plasticity (different learning rates for different weights)
    2. Weight rejuvenation (simulating synaptic turnover)
    3. Weight crystallization (mimicking synapse stabilization)
    4. Multi-synaptic connectivity (weight splitting)
    5. Volume-dependent plasticity (size-based learning rates)

    The converter can operate on both module classes and initialized instances,
    transforming them into biologically-enhanced versions.
    """

    def __init__(self, config: BioConfig = DEFAULT_BIO_CONFIG, **kwargs: Any) -> None:
        """
        Initializes the BioConverter with a configuration.

        Args:
            config: BioConfig object containing parameters for all bio-inspired mechanisms.
                   Defaults to DEFAULT_BIO_CONFIG if not specified.
            **kwargs: Additional keyword arguments to override specific config parameters
                     without creating a new BioConfig object.
        """
        self.config = config._replace(**kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict) -> BioConverter:
        """
        Creates a BioConverter instance from a dictionary of parameters.

        This is a convenience method that allows creating a converter directly
        from a dictionary of parameter values, without manually constructing
        a BioConfig object.

        Args:
            config_dict: Dictionary of parameter names and values.
                        Keys should match BioConfig field names.

        Returns:
            A BioConverter instance with the specified parameters.
        """
        return cls(BioConfig(**config_dict))

    def get_config(self) -> BioConfig:
        """
        Returns the current configuration of the BioConverter.

        Returns:
            The current BioConfig object used by this converter.
        """
        return self.config

    def update_config(self, **kwargs: Any) -> None:
        """
        Updates the configuration of the BioConverter.

        This method allows changing specific configuration parameters after
        the BioConverter has been created. It's useful for incrementally
        adjusting parameters without recreating the entire configuration.

        Args:
            **kwargs: Keyword arguments specifying parameters to update.
                     Keys should match BioConfig field names.
        """
        self.config = self.config._replace(**kwargs)

    def convert(self, module_class_or_instance: Union[Type[nn.Module], nn.Module]) -> Union[Type[nn.Module], nn.Module]:
        """
        Converts a module class or instance by adding bio-inspired modifications.

        This is the main entry point for converting modules. It automatically detects
        whether the input is a class or an instance and dispatches to the appropriate
        conversion method.

        Args:
            module_class_or_instance: The module class or instance to convert.

        Returns:
            The converted module class or instance.

        Raises:
            TypeError: If the input is neither a module class nor an instance.
        """
        if isinstance(module_class_or_instance, nn.Module):
            return self._convert_instance(module_class_or_instance)
        if isinstance(module_class_or_instance, type) and issubclass(module_class_or_instance, nn.Module):
            return self._convert_class(module_class_or_instance)
        raise TypeError(f"Unsupported type for module_class_or_instance: {type(module_class_or_instance)}")

    def _convert_instance(self, module: nn.Module) -> nn.Module:
        """
        Converts an initialized module instance by adding bio-inspired modifications.

        This method:
        1. Marks the last learnable module in the model
        2. Applies BioModule to all eligible layers
        3. Sets layer indices if using layer-adaptive learning rates
        4. Adds BioModule functions to the model instance

        Args:
            module: The initialized module instance to convert.

        Returns:
            The converted module instance with bio-inspired capabilities.
        """
        # Mark the last learnable module to avoid weight splitting on output layers
        self.set_last_module_token_for_model(module)

        # Add BioModule to all eligible layers (Linear, Conv2d)
        module.apply(self._bio_modulize)

        # Set layer indices if using layer-adaptive learning rate strategy
        if self.config.fuzzy_lr_distribution == Distribution.LAYER_ADAPTIVE:
            self._set_layer_indices(module)

        # Add BioModule functions to the model instance
        # This allows calling functions like model.rejuvenate_weights() directly
        for func_name in BioModule.exposed_functions:
            setattr(module, func_name, functools.partial(self._create_instance_method(func_name), module))

        # Add special update_fuzzy_learning_rates method with input handling
        update_method = self._create_update_fuzzy_rates_method()
        setattr(module, 'update_fuzzy_learning_rates', functools.partial(update_method, module))

        return module

    def _create_instance_method(self, func_name: str) -> Callable:
        """
        Creates a method that applies a BioModule function to all submodules.

        This is a helper method that creates functions to be attached to the model,
        which delegate to the corresponding BioModule functions in all submodules.

        Args:
            func_name: The name of the BioModule function to apply.

        Returns:
            A callable that applies the BioModule function to all submodules.
        """

        def instance_method(module):
            def apply_func(_self):
                if hasattr(_self, 'bio_mod'):
                    getattr(_self.bio_mod, func_name)()

            module.apply(apply_func)

        return instance_method

    @staticmethod
    def set_last_module_token_for_model(model):
        """
        Marks the last learnable module in a model to avoid weight splitting on output layers.

        Weight splitting is often undesirable for output layers as it would change the
        output dimensions. This method identifies the last learnable module (Linear or Conv2d)
        and marks it to skip weight splitting.

        Args:
            model: The model to process.
        """
        # Find all learnable modules (Linear or Conv2d)
        learn_modules = []

        for module in model.children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                learn_modules.append(module)

        # Mark the last learnable module if there's more than one
        if len(learn_modules) > 1:
            BioConverter.set_last_module_token_for_module(learn_modules[-1])

    def _convert_class(self, module_class: Type[nn.Module]) -> Type[nn.Module]:
        """
        Converts a module class by adding bio-inspired modifications.

        This method modifies the class's __init__ method to automatically apply
        bio-inspired modifications to any instances created from it. It also adds
        BioModule functions to the class.

        Args:
            module_class: The module class to convert.

        Returns:
            The converted module class with bio-inspired capabilities.

        Raises:
            TypeError: If module_class is not actually a class.
        """
        if not isinstance(module_class, type):
            raise TypeError(f"module_class must be a class; instead got: {type(module_class)}")

        # Helper function to create instance methods that apply to all submodules
        def _apply_to_submodules(method_name: str) -> Callable[[nn.Module], None]:
            def _apply_method(module: nn.Module) -> None:
                if hasattr(module, 'bio_mod'):
                    getattr(module.bio_mod, method_name)()

            return _apply_method

        # Wrapper for the class's __init__ method that applies bio-inspired modifications
        def wrap_init(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapped_init(_self, *args, **kwargs):
                # Call the original __init__
                func(_self, *args, **kwargs)

                # Mark the last learnable module
                self.set_last_module_token_for_model(_self)

                # Add BioModule to all eligible layers
                _self.apply(self._bio_modulize)

            return wrapped_init

        # Save the original __init__ if not already saved
        if not hasattr(module_class, "__inner__init__"):
            module_class.__inner__init__ = module_class.__init__

        # Replace __init__ with our wrapped version
        module_class.__init__ = wrap_init(module_class.__inner__init__)

        # Add BioModule functions to the class
        for func_name in BioModule.exposed_functions:
            setattr(module_class, func_name, lambda self, fn=func_name: self.apply(_apply_to_submodules(fn)))

        return module_class

    def __call__(self, module_class: Type[nn.Module]) -> Type[nn.Module]:
        """
        Makes the BioConverter callable, enabling convenient conversion syntax.

        This allows using the converter as a decorator or in a functional style:
        ```
        converter = BioConverter()
        BioModel = converter(StandardModel)  # Using __call__
        model = BioModel()
        ```

        Args:
            module_class: The module class to convert.

        Returns:
            The converted module class.
        """
        return self.convert(module_class)

    def _bio_modulize(self, module: nn.Module) -> None:
        """
        Adds a BioModule to a PyTorch module if it's an eligible type.

        This method is applied to each module in a model to add bio-inspired
        modifications to eligible layers (Linear, Conv2d).

        Args:
            module: The module to modify.
        """
        if hasattr(module, 'bio_mod'):
            # If module already has a bio_mod, update it with current config
            self._update_bio_mod(module)
        elif isinstance(module, nn.Linear):
            # Handle Linear layers
            self._handle_linear(module)
        elif isinstance(module, nn.Conv2d):
            # Handle Conv2d layers
            self._handle_conv2d(module)

    def _count_learnable_layers(self, model: nn.Module) -> int:
        """
        Counts the number of learnable layers (Linear or Conv2d) in a model.

        This is used for layer-adaptive learning rates to determine
        the total number of layers.

        Args:
            model: The model to analyze.

        Returns:
            The count of learnable layers in the model.
        """
        count = 0
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                count += 1
        return count

    def _set_layer_indices(self, model: nn.Module) -> None:
        """
        Sets layer indices for layer-adaptive learning rates.

        For layer-adaptive learning rates, each layer needs to know its position
        in the network (earlier vs. later layers). This method sets those indices.

        Args:
            model: The model to process.
        """
        if self.config.fuzzy_lr_distribution != Distribution.LAYER_ADAPTIVE:
            return

        # Find all learnable layers
        learnable_layers = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                learnable_layers.append(module)

        total_layers = len(learnable_layers)

        # Set layer index and total layers for each BioModule
        for i, layer in enumerate(learnable_layers):
            if hasattr(layer, 'bio_mod'):
                layer.bio_mod.config = layer.bio_mod.config._replace(
                    fuzzy_lr_layer_index=i,
                    fuzzy_lr_total_layers=total_layers
                )

    def _create_update_fuzzy_rates_method(self):
        """
        Creates the update_fuzzy_learning_rates method for models.

        This method creates a function that updates fuzzy learning rates
        in all BioModules within a model. For activity-dependent rates,
        it can use the input tensor to track neuron activations.

        Returns:
            A callable that applies update_fuzzy_learning_rates to all submodules.
        """

        def update_fuzzy_rates_method(model, x=None):
            """
            Updates fuzzy learning rates for all BioModules in the model.

            Args:
                model: The model to update.
                x: Optional input tensor for activity-dependent distributions.
            """

            def apply_update(_module):
                if hasattr(_module, 'bio_mod'):
                    _module.bio_mod.update_fuzzy_learning_rates(x)

            model.apply(apply_update)

        return update_fuzzy_rates_method

    def _wrap_forward_for_activity_tracking(self, module: nn.Module) -> None:
        """
        Wraps the forward method of a module to track activations for activity-dependent learning.

        This adds activation tracking for modules when using the ACTIVITY distribution
        for fuzzy learning rates, which adjusts learning rates based on neuron activity.

        Args:
            module: The module whose forward method will be wrapped.
        """
        if self.config.fuzzy_lr_distribution != Distribution.ACTIVITY:
            return

        original_forward = module.forward

        # Flag to prevent recursion
        module._tracking_activity = False

        @functools.wraps(original_forward)
        def wrapped_forward(x, *args, **kwargs):
            # Skip updating fuzzy learning rates if we're already tracking
            # This prevents infinite recursion
            if not hasattr(module, '_tracking_activity') or not module._tracking_activity:
                module._tracking_activity = True

                # Do forward pass first
                result = original_forward(x, *args, **kwargs)

                # Then update learning rates
                if hasattr(module, 'bio_mod'):
                    module.bio_mod.update_fuzzy_learning_rates(x)

                module._tracking_activity = False
                return result
            else:
                # If already tracking, just do the forward pass
                return original_forward(x, *args, **kwargs)

        module.forward = wrapped_forward

    def _handle_linear(self, module: nn.Linear) -> None:
        """
        Adds bio-inspired modifications to an nn.Linear module.

        This method:
        1. Validates weight splitting parameters
        2. Adds a BioModule to the Linear layer
        3. Wraps the forward function for weight splitting if needed
        4. Wraps the forward function for activity tracking if needed

        Args:
            module: The nn.Linear module to modify.
        """
        if not hasattr(module, 'bio_mod'):
            # Validate weight splitting parameters
            self._validate_weight_splitting_neurons(self.config.weight_splitting_Gamma, module.in_features)

            # Add BioModule to the layer
            module.add_module('bio_mod', BioModule(lambda: module, config=self.config))

            # Wrap forward function for weight splitting if needed
            # Skip for output layers (marked with last_module_token)
            if self._requires_weight_splitting(module.in_features) and not hasattr(module, "last_module_token"):
                module.forward = self._wrap_forward_with_weight_splitting(module.forward, dim=2)

            # Wrap forward for activity tracking if needed
            self._wrap_forward_for_activity_tracking(module)

    def _handle_conv2d(self, module: nn.Conv2d) -> None:
        """
        Adds bio-inspired modifications to an nn.Conv2d module.

        This method:
        1. Validates weight splitting parameters
        2. Adds a BioModule to the Conv2d layer
        3. Wraps the forward function for weight splitting if needed
        4. Wraps the forward function for activity tracking if needed

        Args:
            module: The nn.Conv2d module to modify.
        """
        if not hasattr(module, 'bio_mod'):
            # Validate weight splitting parameters
            self._validate_weight_splitting_neurons(self.config.weight_splitting_Gamma, module.out_channels)

            # Add BioModule to the layer
            module.add_module('bio_mod', BioModule(lambda: module, config=self.config))

            # Wrap forward function for weight splitting if needed
            # Skip for output layers (marked with last_module_token)
            if self._requires_weight_splitting(module.out_channels) and not hasattr(module, "last_module_token"):
                module.forward = self._wrap_forward_with_weight_splitting(module.forward, dim=4)

            # Wrap forward for activity tracking if needed
            self._wrap_forward_for_activity_tracking(module)

    def _update_bio_mod(self, module: nn.Module) -> None:
        """
        Updates an existing bio_mod in a module with the current configuration.

        This allows reconfiguring modules that already have BioModule attached,
        for example when applying a new converter to an already-converted model.

        Args:
            module: The module containing bio_mod to update.
        """
        module.bio_mod = BioModule(lambda: module, config=self.config)

    def _requires_weight_splitting(self, num_features: int) -> bool:
        """
        Checks if weight splitting is required based on configuration and feature count.

        Weight splitting is applied when:
        1. weight_splitting_Gamma is greater than 1 (splitting is enabled)
        2. The number of features is divisible by weight_splitting_Gamma

        Args:
            num_features: The number of features in the module.

        Returns:
            True if weight splitting should be applied, False otherwise.
        """
        weight_splitting_Gamma = self.config.weight_splitting_Gamma
        return weight_splitting_Gamma > 1 and num_features % weight_splitting_Gamma == 0

    def _wrap_forward_with_weight_splitting(self, forward_func: Callable, dim: int) -> Callable:
        """
        Wraps a forward function with weight splitting functionality.

        Weight splitting implements multi-synaptic connectivity by:
        1. Applying the original forward function
        2. Reshaping outputs to group 'sub-synapses'
        3. Summing across sub-synapses to get combined effects
        4. Repeating (replicating) the result to maintain tensor dimensions

        Args:
            forward_func: The original forward function to wrap.
            dim: The dimension of the input tensor (2 for linear layers, 4 for convolutional layers).

        Returns:
            The wrapped forward function implementing weight splitting.
        """

        # Define weight splitting logic for different tensor dimensions
        def weight_splitting_func(x: torch.Tensor) -> torch.Tensor:
            weight_splitting_Gamma = self.config.weight_splitting_Gamma
            if dim == 2:
                # For 2D tensors (Linear layers)
                assert x.dim() == 2, "Input tensor must be 2D"
                # Reshape to group sub-synapses, sum them, then repeat the result
                return torch.repeat_interleave(
                    x.view(-1, x.size(1) // weight_splitting_Gamma, weight_splitting_Gamma).sum(2),
                    weight_splitting_Gamma, 1)
            elif dim == 4:
                # For 4D tensors (Conv2d layers)
                assert x.dim() == 4, "Input tensor must be 4D"
                # Reshape to group sub-synapses, sum them, then repeat the result
                # Preserve spatial dimensions (H, W)
                return torch.repeat_interleave(
                    x.view(-1, x.size(1) // weight_splitting_Gamma, weight_splitting_Gamma, x.size(-2), x.size(-1)).sum(
                        2), weight_splitting_Gamma, 1)

        # Wrap the original forward function
        @functools.wraps(forward_func)
        def wrapped_forward(*args, **kwargs):
            # First, call the original forward function
            result = forward_func(*args, **kwargs)

            # Apply activation function if configured
            result = self.config.weight_splitting_activation_function(result)

            # Apply weight splitting
            return weight_splitting_func(result)

        return wrapped_forward

    @staticmethod
    def _validate_weight_splitting_neurons(weight_splitting_Gamma: int, num_features: int) -> None:
        """
        Validates that weight splitting parameters are compatible with module dimensions.

        Weight splitting requires that the number of features is divisible by weight_splitting_Gamma
        to ensure clean grouping of sub-synapses.

        Args:
            weight_splitting_Gamma: Number of sub-synapses to create for each connection.
            num_features: Number of features in the module.

        Raises:
            ValueError: If weight_splitting_Gamma > 1 and does not evenly divide num_features.
        """
        if weight_splitting_Gamma > 1 and num_features % weight_splitting_Gamma != 0:
            raise ValueError(
                f"weight_splitting_Gamma ({weight_splitting_Gamma}) must evenly divide the number of features ({num_features}).")

    @staticmethod
    def set_last_module_token_for_module(module: nn.Module) -> nn.Module:
        """
        Marks a module to skip weight splitting.

        This is typically used for output layers where weight splitting
        would change the output dimensions.

        Args:
            module: The module to mark.

        Returns:
            The marked module (for chaining).
        """
        module.last_module_token = True
        return module