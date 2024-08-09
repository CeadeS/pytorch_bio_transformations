from __future__ import annotations

import functools
from typing import Callable, Any, Type, Union

import torch
import torch.nn as nn

from bio_transformations.bio_config import BioConfig, DEFAULT_BIO_CONFIG
from bio_transformations.bio_module import BioModule


class BioConverter:
    """
    A utility class to convert standard PyTorch modules to BioNet modules with bio-inspired modifications.

    This class implements modifications inspired by dendritic spine dynamics observed in our research,
    potentially enhancing the learning and adaptability of artificial neural networks.
    """

    def __init__(self, config: BioConfig = DEFAULT_BIO_CONFIG, **kwargs: Any) -> None:
        """
        Initializes the BioConverter with flexible parameters.

        Args:
            config: BioConfig object containing the parameters.
            **kwargs: Additional keyword arguments to override config parameters.
        """
        self.config = config._replace(**kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict) -> BioConverter:
        """
        Creates a BioConverter instance from a dictionary of parameters.

        Args:
            config_dict: Dictionary of parameter names and values.

        Returns:
            A BioConverter instance with the specified parameters.
        """
        return cls(BioConfig(**config_dict))

    def get_config(self) -> BioConfig:
        """
        Returns the current configuration of the BioConverter.

        Returns:
            The current BioConfig object.
        """
        return self.config

    def update_config(self, **kwargs: Any) -> None:
        """
        Updates the configuration of the BioConverter.

        Args:
            **kwargs: Keyword arguments to update in the configuration.
        """
        self.config = self.config._replace(**kwargs)

    def convert(self, module_class_or_instance: Union[Type[nn.Module], nn.Module]) -> Union[Type[nn.Module], nn.Module]:
        """
        Converts a given module class or instance by adding bio-inspired modifications.

        Args:
            module_class_or_instance: The module class or instance to convert.

        Returns:
            The converted module class or instance.
        """
        if isinstance(module_class_or_instance, nn.Module):
            return self._convert_instance(module_class_or_instance)
        if isinstance(module_class_or_instance, type) and issubclass(module_class_or_instance, nn.Module):
            return self._convert_class(module_class_or_instance)
        raise TypeError(f"Unsupported type for module_class_or_instance: {type(module_class_or_instance)}")

    def _convert_instance(self, module: nn.Module) -> nn.Module:
        """
        Converts an initialized module instance by adding bio-inspired modifications.

        Args:
            module: The initialized module instance to convert.

        Returns:
            The converted module instance.
        """
        self.automark_last_module_for_weight_split_skip(module)
        module.apply(self._bio_modulize)

        # Add BioModule functions to the instance
        for func_name in BioModule.exposed_functions:
            setattr(module, func_name, functools.partial(self._create_instance_method(func_name), module))

        return module

    def _create_instance_method(self, func_name: str) -> Callable:
        """
        Creates a method that applies a BioModule function to all submodules.

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
    def automark_last_module_for_weight_split_skip(model):
        learn_modules = []

        for module in model.children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                learn_modules.append(module)
        if len(learn_modules) > 1:
            BioConverter.mark_skip_weight_splitting(learn_modules[-1])

    def _convert_class(self, module_class: Type[nn.Module]) -> Type[nn.Module]:
        """
        Converts a given module class by adding bio-inspired modifications.

        Args:
            module_class: The module class to convert.

        Returns:
            The converted module class.
        """
        if not isinstance(module_class, type):
            raise TypeError(f"module_class must be a class; instead got: {type(module_class)}")

        def _apply_to_submodules(method_name: str) -> Callable[[nn.Module], None]:
            def _apply_method(module: nn.Module) -> None:
                if hasattr(module, 'bio_mod'):
                    getattr(module.bio_mod, method_name)()

            return _apply_method

        def wrap_init(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapped_init(_self, *args, **kwargs):
                func(_self, *args, **kwargs)
                self.automark_last_module_for_weight_split_skip(_self)
                _self.apply(self._bio_modulize)

            return wrapped_init

        if not hasattr(module_class, "__inner__init__"):
            module_class.__inner__init__ = module_class.__init__

        module_class.__init__ = wrap_init(module_class.__inner__init__)

        for func_name in BioModule.exposed_functions:
            setattr(module_class, func_name, lambda self, fn=func_name: self.apply(_apply_to_submodules(fn)))

        return module_class

    def __call__(self, module_class: Type[nn.Module]) -> Type[nn.Module]:
        """
        Makes the BioConverter callable, allowing for convenient conversion of module classes.
        """
        return self.convert(module_class)

    def _bio_modulize(self, module: nn.Module) -> None:
        """
        Adds bio-inspired modifications to a module.

        Args:
            module: The module to modify.
        """
        if hasattr(module, 'bio_mod'):
            self._update_bio_mod(module)
        elif isinstance(module, nn.Linear):
            self._handle_linear(module)
        elif isinstance(module, nn.Conv2d):
            self._handle_conv2d(module)

    def _handle_linear(self, module: nn.Linear) -> None:
        """
        Adds bio-inspired modifications to an nn.Linear module.
        """
        if not hasattr(module, 'bio_mod'):
            self._validate_weight_splitting_neurons(self.config.weight_splitting_Gamma, module.in_features)
            module.add_module('bio_mod', BioModule(lambda: module, config=self.config))
            if self._requires_weight_splitting(module.in_features) and not hasattr(module, "weight_splitting_skip"):
                module.forward = self._wrap_forward_with_weight_splitting(module.forward, dim=2)

    def _handle_conv2d(self, module: nn.Conv2d) -> None:
        """
        Adds bio-inspired modifications to an nn.Conv2d module.
        """
        if not hasattr(module, 'bio_mod'):
            self._validate_weight_splitting_neurons(self.config.weight_splitting_Gamma, module.out_channels)
            module.add_module('bio_mod', BioModule(lambda: module, config=self.config))
            if self._requires_weight_splitting(module.out_channels) and not hasattr(module, "weight_splitting_skip"):
                module.forward = self._wrap_forward_with_weight_splitting(module.forward, dim=4)

    def _update_bio_mod(self, module: nn.Module) -> None:
        """
        Updates an existing bio_mod in the module.

        Args:
            module: The module containing bio_mod to update.
        """
        module.bio_mod = BioModule(lambda: module, config=self.config)

    def _requires_weight_splitting(self, num_features: int) -> bool:
        """
        Checks if weight_splitting is required based on the number of features.

        Args:
            num_features: The number of features in the module.

        Returns:
            True if weight_splitting is required, False otherwise.
        """
        weight_splitting_Gamma = self.config.weight_splitting_Gamma
        return weight_splitting_Gamma > 1 and num_features % weight_splitting_Gamma == 0

    def _wrap_forward_with_weight_splitting(self, forward_func: Callable, dim: int) -> Callable:
        """
        Wraps the forward function with the weight_splitting step.

        Args:
            forward_func: The original forward function.
            dim: The dimension of the input tensor (2 for linear layers, 4 for convolutional layers).

        Returns:
            The wrapped forward function with weight_splitting.
        """

        def weight_splitting_func(x: torch.Tensor) -> torch.Tensor:
            weight_splitting_Gamma = self.config.weight_splitting_Gamma
            if dim == 2:
                assert x.dim() == 2, "Input tensor must be 2D"
                return torch.repeat_interleave(
                    x.view(-1, x.size(1) // weight_splitting_Gamma, weight_splitting_Gamma).sum(2),
                    weight_splitting_Gamma, 1)
            elif dim == 4:
                assert x.dim() == 4, "Input tensor must be 4D"
                return torch.repeat_interleave(
                    x.view(-1, x.size(1) // weight_splitting_Gamma, weight_splitting_Gamma, x.size(-2), x.size(-1)).sum(
                        2), weight_splitting_Gamma, 1)

        @functools.wraps(forward_func)
        def wrapped_forward(*args, **kwargs):
            result = forward_func(*args, **kwargs)
            result = self.config.weight_splitting_activation_function(result)
            return weight_splitting_func(result)

        return wrapped_forward

    @staticmethod
    def _validate_weight_splitting_neurons(weight_splitting_Gamma: int, num_features: int) -> None:
        """
        Validates that the number of weight_splitting neurons is appropriate.

        Args:
            weight_splitting_Gamma: Number of neurons to weight_splitting.
            num_features: Number of features in the module.

        Raises:
            ValueError: If weight_splitting_Gamma is greater than 1 and does not evenly divide num_features.
        """
        if weight_splitting_Gamma > 1 and num_features % weight_splitting_Gamma != 0:
            raise ValueError(
                f"weight_splitting_Gamma ({weight_splitting_Gamma}) must evenly divide the number of features ({num_features}).")

    @staticmethod
    def mark_skip_weight_splitting(module: nn.Module) -> nn.Module:
        """
        Marks a module to skip weight_splitting.

        Args:
            module: The module to mark.

        Returns:
            The marked module.
        """
        module.weight_splitting_skip = True
        return module
