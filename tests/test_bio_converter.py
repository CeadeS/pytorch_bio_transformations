import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from bio_transformations import BioConverter, BioModule
from bio_transformations.bio_config import BioConfig, DEFAULT_BIO_CONFIG


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(26 * 26 * 2, 10)  # assuming input image size is 28x28

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc1(x)


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.constant_(module.weight, 0.5)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.1)


def test_bioconverter_default_config():
    converter = BioConverter()
    assert converter.get_config() == DEFAULT_BIO_CONFIG


def test_bioconverter_custom_config():
    custom_config = BioConfig(fuzzy_learning_rate_factor_nu=0.2, dampening_factor=0.7)
    converter = BioConverter(config=custom_config)
    assert converter.get_config() == custom_config


def test_bioconverter_from_dict():
    config_dict = {'fuzzy_learning_rate_factor_nu': 0.2, 'dampening_factor': 0.7}
    converter = BioConverter.from_dict(config_dict)
    assert converter.get_config().fuzzy_learning_rate_factor_nu == 0.2
    assert converter.get_config().dampening_factor == 0.7


def test_bioconverter_update_config():
    converter = BioConverter()
    converter.update_config(crystal_thresh=5e-05)
    assert converter.get_config().crystal_thresh == 5e-05


def test_biomodule_with_cnn():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)

    custom_config = BioConfig(fuzzy_learning_rate_factor_nu=0.16, dampening_factor=0.6, crystal_thresh=4.5e-05,
                              rejuvenation_parameter_dre=8.0, weight_splitting_Gamma=2, apply_dales_principle=False)
    converter = BioConverter(config=custom_config)
    model = converter(SimpleCNN)()
    model.apply(initialize_weights)

    input_tensor = torch.ones((1, 1, 28, 28))
    target_tensor = torch.tensor([1], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    output = model(input_tensor)
    loss = criterion(output, target_tensor)

    optimizer.zero_grad()
    loss.backward()

    #assert f"{model.conv1.weight.grad.data.max():.4e}" == "1.2609e-04"

    model.conv1.weight.data[0, 0, 0, 0] = 0.01
    model.rejuvenate_weights()
    #assert f"{model.conv1.weight.data.abs().min():1.4f}" == "0.2570"
    model.fuzzy_learning_rates()

    #ssert f"{model.conv1.weight.grad.data.max():.4e}" == "1.4158e-04"

    optimizer.step()

    output_after = model(input_tensor)
    #assert f"{output_after.mean().item():5.4f}" == "5707.5200"
    assert output_after is not None, "Output after BioModule operations is None"

    print("CNN Test Passed")


def test_biomodule_with_mlp():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)

    custom_config = BioConfig(fuzzy_learning_rate_factor_nu=0.16, dampening_factor=0.6, crystal_thresh=4.5e-05,
                              rejuvenation_parameter_dre=8.0, weight_splitting_Gamma=2, apply_dales_principle=False)
    converter = BioConverter(config=custom_config)
    model = converter(SimpleMLP)()
    model.apply(initialize_weights)

    input_tensor = torch.ones((1, 20))
    target_tensor = torch.tensor([1], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    output = model(input_tensor)
    loss = criterion(output, target_tensor)

    optimizer.zero_grad()
    loss.backward()

    #assert f"{model.fc1.weight.grad.data.max():.4e}" == "5.9605e-08"

    model.fc1.weight.data[0, 0] = 0.01
    model.rejuvenate_weights()
    #assert f"{model.fc1.weight.data.abs().min():1.4f}" == "0.1811"
    model.fuzzy_learning_rates()

    #assert f"{model.fc1.weight.grad.data.max():.4e}" == "6.9026e-08"

    optimizer.step()

    output_after = model(input_tensor)
    #assert f"{output_after.mean().item():4.4f}" == "504.5995"
    #assert output_after is not None, "Output after BioModule operations is None"

    print("MLP Test Passed")


def test_convert_module_class():
    custom_config = BioConfig(fuzzy_learning_rate_factor_nu=0.16, dampening_factor=0.6, crystal_thresh=4.5e-05,
                              rejuvenation_parameter_dre=8.0, weight_splitting_Gamma=2, apply_dales_principle=False)
    converter = BioConverter(config=custom_config)

    ConvertedMLP = converter.convert(SimpleMLP)
    converted_model = ConvertedMLP()

    assert hasattr(converted_model.fc1, 'bio_mod')
    assert isinstance(converted_model.fc1.bio_mod, BioModule)
    assert converted_model.fc1.bio_mod.config.fuzzy_learning_rate_factor_nu == custom_config.fuzzy_learning_rate_factor_nu
    assert converted_model.fc1.bio_mod.config.dampening_factor == custom_config.dampening_factor
    assert converted_model.fc1.bio_mod.config.crystal_thresh == custom_config.crystal_thresh
    assert converted_model.fc1.bio_mod.config.rejuvenation_parameter_dre == custom_config.rejuvenation_parameter_dre
    assert converted_model.fc1.bio_mod.config.weight_splitting_Gamma == custom_config.weight_splitting_Gamma
    assert converted_model.fc1.bio_mod.config.apply_dales_principle == custom_config.apply_dales_principle


def test_convert_initialized_module_instance():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)

    custom_config = BioConfig(fuzzy_learning_rate_factor_nu=0.10, dampening_factor=0.3, crystal_thresh=4.e-05,
                              rejuvenation_parameter_dre=6.0, weight_splitting_Gamma=1, apply_dales_principle=False)
    converter = BioConverter(config=custom_config)

    model = SimpleMLP()
    converted_model = converter.convert(model)

    assert hasattr(converted_model.fc1, 'bio_mod')
    assert isinstance(converted_model.fc1.bio_mod, BioModule)
    assert converted_model.fc1.bio_mod.config.fuzzy_learning_rate_factor_nu == custom_config.fuzzy_learning_rate_factor_nu
    assert converted_model.fc1.bio_mod.config.dampening_factor == custom_config.dampening_factor
    assert converted_model.fc1.bio_mod.config.crystal_thresh == custom_config.crystal_thresh
    assert converted_model.fc1.bio_mod.config.rejuvenation_parameter_dre == custom_config.rejuvenation_parameter_dre
    assert converted_model.fc1.bio_mod.config.weight_splitting_Gamma == custom_config.weight_splitting_Gamma
    assert converted_model.fc1.bio_mod.config.apply_dales_principle == custom_config.apply_dales_principle

    # Test updating existing Model with new converter
    new_config = BioConfig(fuzzy_learning_rate_factor_nu=0.16, dampening_factor=0.6, crystal_thresh=4.5e-05,
                           rejuvenation_parameter_dre=8.0, weight_splitting_Gamma=2, apply_dales_principle=False)
    new_converter = BioConverter(config=new_config)

    converted_model = new_converter.convert(converted_model)

    assert isinstance(converted_model.fc1.bio_mod, BioModule)
    assert converted_model.fc1.bio_mod.config.fuzzy_learning_rate_factor_nu == new_config.fuzzy_learning_rate_factor_nu
    assert converted_model.fc1.bio_mod.config.dampening_factor == new_config.dampening_factor
    assert converted_model.fc1.bio_mod.config.crystal_thresh == new_config.crystal_thresh
    assert converted_model.fc1.bio_mod.config.rejuvenation_parameter_dre == new_config.rejuvenation_parameter_dre
    assert converted_model.fc1.bio_mod.config.weight_splitting_Gamma == new_config.weight_splitting_Gamma
    assert converted_model.fc1.bio_mod.config.apply_dales_principle == new_config.apply_dales_principle

    model = converted_model
    model.apply(initialize_weights)

    input_tensor = torch.ones((1, 20))
    target_tensor = torch.tensor([1], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    output = model(input_tensor)
    loss = criterion(output, target_tensor)

    optimizer.zero_grad()
    loss.backward()

    #assert f"{model.fc1.weight.grad.data.max():.4e}" == "5.9605e-08"

    model.fc1.weight.data[0, 0] = 0.01
    model.rejuvenate_weights()
    #assert f"{model.fc1.weight.data.abs().min():1.4f}" == "0.0425"
    model.fuzzy_learning_rates()

    #assert f"{model.fc1.weight.grad.data.max():.4e}" == "6.9066e-08"

    optimizer.step()

    output_after = model(input_tensor)
    #assert f"{output_after.mean().item():4.4f}" == "504.8892"
    assert output_after is not None, "Output after BioModule operations is None"


def test_converter_no_class():
    converter = BioConverter()
    with pytest.raises(TypeError, match=f"module_class must be a class; instead got: {type('invalid class')}"):
        converter._convert_class("invalid class")


def test_invalid_module_conversion():
    converter = BioConverter()
    with pytest.raises(TypeError, match=f"Unsupported type for module_class_or_instance: {type(torch.zeros(42))}"):
        converter.convert(torch.zeros(42))


def test_weight_splitting():
    class TestLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(6, 4, bias=False)

        def forward(self, x):
            return self.linear(x)

    class TestConv2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(6, 4, kernel_size=3, padding=1, bias=False)

        def forward(self, x):
            return self.conv(x)

    # Test for Linear layer (dim=2)
    config = BioConfig(weight_splitting_Gamma=2)
    converter = BioConverter(config=config)

    linear_model = TestLinear()
    converted_linear = converter.convert(TestLinear())

    converted_linear.linear.weight.data = linear_model.linear.weight.data

    input_2d = torch.ones(2, 6)
    output_2d = converted_linear(input_2d)

    assert output_2d.shape == (2, 4), f"Expected output shape (2, 4), but got {output_2d.shape}"

    # Verify weight splitting for Linear layer
    original_output = linear_model(input_2d)
    expected_output = torch.repeat_interleave(original_output.view(-1, 2, 2).sum(2), 2, 1)

    assert torch.allclose(output_2d, expected_output,
                          atol=1e-5), "Weight splitting for Linear layer not working as expected"

    # Test for Conv2d layer (dim=4)
    conv_model = TestConv2d()
    converted_conv = converter.convert(TestConv2d())

    conv_model.conv.weight.data = converted_conv.conv.weight.data

    input_4d = torch.randn(2, 6, 8, 8)
    output_4d = converted_conv(input_4d)

    assert output_4d.shape == (2, 4, 8, 8), f"Expected output shape (2, 4, 8, 8), but got {output_4d.shape}"

    # Verify weight splitting for Conv2d layer
    original_output = conv_model(input_4d)
    expected_output = torch.repeat_interleave(original_output.view(-1, 2, 2, 8, 8).sum(2), 2, 1)
    assert torch.allclose(output_4d, expected_output,
                          atol=1e-5), "Weight splitting for Conv2d layer not working as expected"

    # Test with non-divisible number of features
    class InvalidLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 3)  # 5 is not divisible by weight_splitting_Gamma=2

        def forward(self, x):
            return self.linear(x)

    InvalidLinear()(torch.rand(2, 5))

    with pytest.raises(ValueError, match="weight_splitting_Gamma .* must evenly divide the number of features"):
        converter.convert(InvalidLinear())

    # Test weight splitting activation function
    def custom_activation(x):
        return x * 2

    config_with_activation = BioConfig(weight_splitting_Gamma=2, weight_splitting_activation_function=custom_activation)
    converter_with_activation = BioConverter(config=config_with_activation)

    linear_model_with_activation = TestLinear()
    converted_linear_with_activation = converter_with_activation.convert(linear_model_with_activation)

    converted_linear_with_activation.linear.weight.data = converted_linear.linear.weight.data

    input_2d = torch.ones(2, 6)
    output_2d_with_activation = converted_linear_with_activation(input_2d)
    output_2d_without_activation = converted_linear(input_2d)

    assert torch.allclose(output_2d_with_activation, output_2d_without_activation * 2,
                          atol=1e-5), "Custom activation function not applied correctly"


def test_mlp_deterministic():
    model = nn.Sequential(nn.Linear(4, 2, bias=False), nn.ReLU())
    model[0].weight.data = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=model[0].weight.data.dtype).view(
        model[0].weight.data.shape)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    converter = BioConverter(BioConfig(weight_splitting_Gamma=0, rejuvenation_parameter_dre=3))
    model = converter(model)
    model[0].bio_mod.fuzzy_learning_rate_parameters.data = torch.ones_like(
        model[0].bio_mod.fuzzy_learning_rate_parameters.data) * 1.1

    input = torch.FloatTensor([[0, 0, 0, 1]])
    result = model(input)
    target = torch.LongTensor([[3, 6]])
    loss = torch.nn.L1Loss()(result, target)

    optimizer.zero_grad()
    loss.backward()

    sum_before = model[0].weight.data.sum()
    model.rejuvenate_weights()
    assert model[0].weight.data.sum() != sum_before
    model.fuzzy_learning_rates()

    #assert f"{model[0].weight.grad.data.max():.1e}" == "5.5e-01"

    optimizer.step()


def test_instance_conversion():
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 5)

        def forward(self, x):
            x = F.relu(self.linear1(x))
            return self.linear2(x)

    # Create an instance of the model
    model = TestModel()
    model(torch.rand(10, 10))
    torch.nn.L1Loss()(model(torch.rand(10, 10)), torch.rand(10, 1)).backward()

    # Convert the instance
    converter = BioConverter(config=BioConfig(weight_splitting_Gamma=2))
    converted_model = converter.convert(model)

    # Check if BioModule functions are added to the instance
    assert hasattr(converted_model, 'rejuvenate_weights'), "rejuvenate_weights not added to the instance"
    assert hasattr(converted_model, 'crystallize'), "crystallize not added to the instance"
    assert hasattr(converted_model, 'fuzzy_learning_rates'), "fuzzy_learning_rates not added to the instance"
    assert hasattr(converted_model, 'volume_dependent_lr'), "volume_dependent_lr not added to the instance"

    # Test if the functions can be called without errors
    converted_model.rejuvenate_weights()

    converted_model.crystallize()
    converted_model.fuzzy_learning_rates()
    converted_model.volume_dependent_lr()

    # Test if weight splitting is applied
    input_tensor = torch.randn(10, 10)
    output = converted_model(input_tensor)
    assert output.shape == (10, 5), f"Expected output shape (10, 5), but got {output.shape}"

    # Test if weight splitting is applied to both layers
    assert converted_model.linear1.weight.shape == (
        20, 10), f"Expected linear1 weight shape (20, 10), but got {converted_model.linear1.weight.shape}"
    assert converted_model.linear2.weight.shape == (
        5, 20), f"Expected linear2 weight shape (5, 20), but got {converted_model.linear2.weight.shape}"
