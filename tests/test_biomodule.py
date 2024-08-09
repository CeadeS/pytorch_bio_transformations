import pytest
import torch
import torch.nn as nn

from bio_transformations.bio_config import BioConfig, DEFAULT_BIO_CONFIG
from bio_transformations.bio_module import BioModule


def create_linear_layer():
    linear_layer = nn.Linear(10, 10)
    linear_layer.weight.requires_grad = True
    return linear_layer


def test_biomodule_invalid_parent():
    with pytest.raises(AssertionError, match="parent must be a Callable returning an nn.Module instance"):
        BioModule("invalid parent")

    with pytest.raises(AssertionError, match="parent output must be an instance of nn.Module"):
        BioModule(lambda: "not an nn.Module")


def test_dales_principle():
    linear = create_linear_layer()
    custom_config = BioConfig(fuzzy_learning_rate_factor_nu=0.2, dampening_factor=0.7, crystal_thresh=5e-05,
                              rejuvenation_parameter_dre=7.0)
    bio_mod = BioModule(lambda: linear, config=custom_config)
    with pytest.raises(AttributeError, match=f"Can not use dalians network initialization on {type(torch.rand((10)))}"):
        BioModule.dalian_network_initialization(torch.rand((10)))
    with pytest.raises(AttributeError,
                       match=f"Can not enforce dales principle without apply_dales_principle set True."):
        bio_mod.enforce_dales_principle()


def test_biomodule_default_config():
    linear = create_linear_layer()
    bio_mod = BioModule(lambda: linear)
    assert bio_mod.config == DEFAULT_BIO_CONFIG


def test_biomodule_custom_config():
    linear = create_linear_layer()
    custom_config = BioConfig(fuzzy_learning_rate_factor_nu=0.2, dampening_factor=0.7, crystal_thresh=5e-05,
                              rejuvenation_parameter_dre=7.0)
    bio_mod = BioModule(lambda: linear, config=custom_config)
    assert bio_mod.config == custom_config


def test_volume_dependent_lr():
    linear = nn.Linear(4, 4)

    with torch.no_grad():
        linear.weight.data = torch.linspace(0.1, 1.0, 16).reshape(4, 4)
        linear.weight.grad = torch.ones_like(linear.weight)

    bio_mod = BioModule(lambda: linear, config=BioConfig(base_lr=0.1, stability_factor=2.0, lr_variability=0.2))

    bio_mod.volume_dependent_lr()

    assert not torch.allclose(linear.weight.grad, torch.ones_like(linear.weight)), "Gradients should be modified"

    large_weights = linear.weight.data > 0.5
    small_weights = linear.weight.data <= 0.5
    large_lr = linear.weight.grad[large_weights].abs().mean()
    small_lr = linear.weight.grad[small_weights].abs().mean()
    assert large_lr < small_lr, "Larger weights should have smaller learning rates on average"

    old_grads = linear.weight.grad.clone()
    bio_mod.volume_dependent_lr()
    new_grads = linear.weight.grad.clone()
    assert not torch.allclose(new_grads, old_grads), "Learning rates should be stochastic"

    assert torch.all(linear.weight.grad > 0), "All learning rates should be positive"

    assert torch.all(linear.weight.grad < bio_mod.config.base_lr * 2), "Learning rates should not be too large"

    results = []
    for _ in range(3):
        bio_mod.volume_dependent_lr()
        results.append(linear.weight.grad.clone())

    assert all(not torch.allclose(results[i], results[j]) for i in range(len(results)) for j in
               range(i + 1, len(results))), "Repeated calls should produce different results"

    bio_mod.get_parent().weight.grad = None
    with pytest.raises(RuntimeError, match="No gradients found for the weights"):
        bio_mod.volume_dependent_lr()

    print("All tests for volume_dependent_lr passed successfully!")


def test_biomodule_invalid_parameters():
    with pytest.raises(AssertionError, match="fuzzy_learning_rate_factor_nu must be positive"):
        BioModule(lambda: nn.Linear(10, 10), config=BioConfig(fuzzy_learning_rate_factor_nu=-0.1))

    with pytest.raises(AssertionError, match="dampening_factor must be between 0 and 1"):
        BioModule(lambda: nn.Linear(10, 10), config=BioConfig(dampening_factor=1.5))

    with pytest.raises(AssertionError, match="crystal_thresh must be positive"):
        BioModule(lambda: nn.Linear(10, 10), config=BioConfig(crystal_thresh=-0.1))

    with pytest.raises(AssertionError, match="rejuvenation_parameter_dre must be positive"):
        BioModule(lambda: nn.Linear(10, 10), config=BioConfig(rejuvenation_parameter_dre=-1.0))

    with pytest.raises(AssertionError, match="weight_splitting_Gamma cannot be negative"):
        BioModule(lambda: nn.Linear(10, 10), config=BioConfig(weight_splitting_Gamma=-1))


def test_rejuvenate_weights():
    linear_layer = create_linear_layer()
    bio_mod = BioModule(lambda: linear_layer)

    initial_weights = linear_layer.weight.data.clone()
    bio_mod.rejuvenate_weights()
    assert not torch.allclose(linear_layer.weight.data,
                              initial_weights), "Weights should be modified after rejuvenation"

    linear_layer.weight.requires_grad = False
    with pytest.raises(RuntimeError, match="Weights do not require gradients"):
        bio_mod.rejuvenate_weights()

    linear_layer.weight.requires_grad = True
    with pytest.raises(RuntimeError, match="Error computing rejuvenation threshold."):
        linear_layer.weight.data[0, 0] *= torch.nan
        bio_mod.rejuvenate_weights()


def test_crystallize():
    linear_layer = create_linear_layer()
    bio_mod = BioModule(lambda: linear_layer)

    linear_layer.weight.grad = torch.ones_like(linear_layer.weight)
    initial_fuzzy_lr = bio_mod.fuzzy_learning_rate_parameters.clone()
    bio_mod.crystallize()
    assert not torch.allclose(bio_mod.fuzzy_learning_rate_parameters,
                              initial_fuzzy_lr), "Fuzzy learning rates should be modified after crystallization"

    linear_layer.weight.requires_grad = False
    with pytest.raises(RuntimeError, match="Weights do not require gradients"):
        bio_mod.crystallize()


def test_fuzzy_learning_rates():
    linear_layer = create_linear_layer()
    bio_mod = BioModule(lambda: linear_layer)

    linear_layer.weight.grad = torch.ones_like(linear_layer.weight)
    initial_grad = linear_layer.weight.grad.clone()
    bio_mod.fuzzy_learning_rates()
    assert torch.allclose(linear_layer.weight.grad,
                          initial_grad * bio_mod.fuzzy_learning_rate_parameters), "Gradients should be scaled by fuzzy learning rates"

    linear_layer = create_linear_layer()
    linear_layer.weight.grad = None
    with pytest.raises(RuntimeError, match="No gradients found for the weights"):
        bio_mod.fuzzy_learning_rates()


def test_enforce_dales_principle():
    linear_layer = create_linear_layer()
    bio_mod = BioModule(lambda: linear_layer, config=BioConfig(apply_dales_principle=True))

    bio_mod.get_parent().weight.data = bio_mod.sign * bio_mod.get_parent().weight.data * -1
    bio_mod.enforce_dales_principle()
    assert torch.sum((
                             bio_mod.sign * bio_mod.get_parent().weight.data) < 0) == 0, "All weights should have the same sign as their corresponding sign parameter"

    delattr(bio_mod, 'sign')
    with pytest.raises(AttributeError, match="sign attribute not found"):
        bio_mod.enforce_dales_principle()


def test_l1_reg():
    linear_layer = create_linear_layer()
    bio_mod = BioModule(lambda: linear_layer)

    l1_reg = bio_mod.l1_reg()
    assert isinstance(l1_reg, torch.Tensor), "L1 regularization should return a tensor"
    assert l1_reg.ndim == 0, "L1 regularization should be a scalar"
    assert l1_reg.item() >= 0, "L1 regularization should be non-negative"
