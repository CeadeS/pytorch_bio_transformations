from typing import NamedTuple, Callable, Any, Literal
from enum import Enum
import torch.nn as nn


class Distribution(str, Enum):
    BASELINE = "baseline"
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    GAMMA = "gamma"
    BETA = "beta"
    LAYER_ADAPTIVE = "layer_adaptive"
    WEIGHT_ADAPTIVE = "weight_adaptive"
    TEMPORAL = "temporal"
    ACTIVITY = "activity"


class BioConfig(NamedTuple):
    # Existing parameters
    weight_splitting_activation_function: Callable[[Any], Any] = nn.Identity()
    fuzzy_learning_rate_factor_nu: float = 0.16
    dampening_factor: float = 0.6
    crystal_thresh: float = 4.5e-05
    rejuvenation_parameter_dre: float = 8.0
    weight_splitting_Gamma: int = 0
    apply_dales_principle: bool = False
    base_lr: float = 0.1
    stability_factor: float = 2.0
    lr_variability: float = 0.2

    # parameters for fuzzy learning rate distributions
    fuzzy_lr_distribution: Distribution = Distribution.UNIFORM
    fuzzy_lr_min: float = 0.5  # Minimum learning rate multiplier
    fuzzy_lr_max: float = 2.0  # Maximum learning rate multiplier
    fuzzy_lr_dynamic: bool = False  # Whether to use dynamic updates
    fuzzy_lr_decay: float = 0.9995  # Decay rate for dynamic updates
    fuzzy_lr_update_freq: int = 100  # Update frequency for dynamic rates
    fuzzy_lr_layer_index: int = -1  # Layer index for layer-adaptive (set by converter)
    fuzzy_lr_total_layers: int = -1  # Total layers for layer-adaptive (set by converter)


DEFAULT_BIO_CONFIG = BioConfig()