from typing import NamedTuple, Callable, Any

import torch.nn as nn


class BioConfig(NamedTuple):
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


DEFAULT_BIO_CONFIG = BioConfig()
