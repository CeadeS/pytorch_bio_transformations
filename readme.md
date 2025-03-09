[![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/mit)
![PyPI](https://img.shields.io/pypi/v/pytorch_bio_transformations)
[![tests](https://github.com/CeadeS/pytorch_bio_transformations/actions/workflows/test.yml/badge.svg)](https://github.com/CeadeS/pytorch_bio_transformations/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/CeadeS/pytorch_bio_transformations/branch/dev/graph/badge.svg?token=I11PUI5K0S)](https://codecov.io/gh/CeadeS/pytorch_bio_transformations)
[![Build and deploy docs](https://github.com/CeadeS/pytorch_bio_transformations/actions/workflows/documentation.yml/badge.svg)](https://github.com/CeadeS/pytorch_bio_transformations/actions/workflows/documentation.yml)
[![Release](https://github.com/CeadeS/pytorch_bio_transformations/actions/workflows/release_and_deploy.yml/badge.svg)](https://github.com/CeadeS/pytorch_bio_transformations/actions/workflows/release_and_deploy.yml)

# PyTorch Bio Transformations

Please visit the [Documentation](https://ceades.github.io/pytorch_bio_transformations/index.html) for further information or refer to the [Publication](#publication)

## Table of Contents
1. [Project Description](#project-description)
2. [Getting Started](#getting-started)
3[Key Features](#key-features)
4[Installation Instructions](#installation-instructions)
5[Usage](#usage)
6[Advanced Usage](#advanced-usage)
7[Extending Functionality](#extending-functionality)
8[Contributing Guidelines](#contributing-guidelines)
9[License Information](#license-information)
10[Publication](#publication)

## Project Description

PyTorch Bio Transformations is a Python library that implements biologically inspired modifications to artificial neural networks, based on research on dendritic spine dynamics. It aims to explore and enhance the learning capabilities of neural networks by mimicking the plasticity and stability characteristics observed in biological synapses.

This project is primarily targeted at researchers and developers in the fields of machine learning and computational neuroscience who are interested in exploring bio-inspired approaches to augment neural network performance.

## Getting Started

```bash
# Install the package
pip install pytorch_bio_transformations

# Convert your PyTorch model in just 3 lines
from bio_transformations import BioConverter
converter = BioConverter()
bio_model = converter(your_pytorch_model)

# Use bio_model as you would a regular PyTorch model
# During training, apply bio-inspired mechanisms
optimizer.zero_grad()
loss.backward()
bio_model.fuzzy_learning_rates()  # Apply diverse learning rates
bio_model.crystallize()           # Stabilize well-optimized weights
optimizer.step()
```

## Key Features

Bio Transformations implements several biologically inspired methods, each mimicking specific aspects of neuronal behavior:

1. **Synaptic Diversity** (`fuzzy_learning_rates`): Implements diverse learning rates for different "synapses" (weights), mimicking the variability observed in biological synapses.

2. **Structural Plasticity** (`rejuvenate_weights`): Simulates spine turnover by randomly reinitializing certain weights, allowing for the "formation" of new connections and the "pruning" of others.

3. **Synaptic Stabilization** (`crystallize`): Mimics the stabilization of frequently used synapses by reducing learning rates for well-optimized weights.

4. **Multi-synaptic Connectivity** (`weight_splitting`): Allows multiple "synapses" (sub-weights) to exist for each connection, enhancing the reliability and flexibility of neural circuits.

5. **Volume-dependent Plasticity** (`volume_dependent_lr`): Adjusts learning rates based on weight magnitude (analogous to spine volume), where larger weights have smaller, less variable learning rates.

6. **Homeostatic Plasticity** (`scale_grad`): Implements synaptic scaling to maintain overall network stability while allowing for learning.

7. **Dale's Principle** (`enforce_dales_principle`): Ensures that all outgoing weights from a given artificial "neuron" have the same sign, mimicking the constraints imposed by neurotransmitter types.

These methods work in concert to create a learning process that more closely resembles the dynamics observed in biological neural networks, potentially leading to improved learning and generalization in artificial neural networks.

## Installation Instructions

You can install Bio Transformations using pip or from source.
### Install PyTorch

#### You can install PyTorch on Linux with pip:
GPU/CUDA12.4: 
```bash
pip3 install torch torchvision torchaudio
```
CPU: 
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### On Windows with pip
GPU/CUDA12.4: 
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
CPU: 
```bash
pip3 install torch torchvision torchaudio
```

#### On Mac with pip

```bash
pip3 install torch torchvision torchaudio
```
### Installing pytorch_bio_transfomrations
#### Option 1: Using pip (Simplest Method)

```bash
pip install pytorch_bio_transformations
```

#### Option 2: From Source (For Development or Latest Changes)

```bash
git clone https://github.com/CeadeS/pytorch_bio_transformations
cd pytorch_bio_transformations
pip install -r requirements.txt
pip install -e .
```

#### Verifying Installation

```bash
python -c "import bio_transformations; print(bio_transformations.__version__)"
```

## Usage

### Basic Usage Example

```python
import torch
import torch.nn as nn
from bio_transformations import BioConverter, BioConfig

# Define your model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Create and convert your model
model = SimpleModel()
converter = BioConverter(
    fuzzy_learning_rate_factor_nu=0.16,  # Controls the diversity in learning rates
    dampening_factor=0.6,                # Controls the stability increase during crystallization
    crystal_thresh=4.5e-05               # Threshold for identifying weights to crystallize
)
bio_model = converter(model)

# Use bio_model as you would a regular PyTorch model
x = torch.randn(1, 10)
output = bio_model(x)
print(output)
```

### Training Example with Bio-inspired Mechanisms

```python
import torch
import torch.nn as nn
import torch.optim as optim
from bio_transformations import BioConverter

# Define a simple model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Convert the model to use bio-inspired mechanisms
converter = BioConverter(
    fuzzy_learning_rate_factor_nu=0.16,  # Controls variability in learning rates
    crystal_thresh=4.5e-05,              # Threshold for synapse crystallization
    rejuvenation_parameter_dre=8.0       # Controls the rate of weight rejuvenation
)
bio_model = converter(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(bio_model.parameters(), lr=0.01)

# Example training loop
def train(data_loader, epochs=5):
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            # Forward pass
            outputs = bio_model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Apply bio-inspired mechanisms
            bio_model.volume_dependent_lr()   # Adjust learning rates based on weight size
            bio_model.fuzzy_learning_rates()  # Apply diverse learning rates
            bio_model.crystallize()           # Stabilize well-optimized weights
            
            # Update weights
            optimizer.step()
            
        # Periodically apply weight rejuvenation (e.g., once per epoch)
        bio_model.rejuvenate_weights()
        
        print(f"Epoch {epoch+1}/{epochs} completed")
```

## Advanced Usage

### Configuration Options

Bio Transformations offers extensive configuration options through the `BioConfig` class:

```python
from bio_transformations import BioConverter, BioConfig
from bio_transformations.bio_config import Distribution

# Create a detailed configuration
config = BioConfig(
    # Fuzzy learning rate parameters
    fuzzy_learning_rate_factor_nu=0.16,     # Controls the variability in learning rates
    fuzzy_lr_distribution=Distribution.NORMAL,  # Distribution strategy for learning rates
    fuzzy_lr_dynamic=True,                  # Whether to update learning rates during training
    
    # Synaptic stabilization parameters
    dampening_factor=0.6,                   # Factor for reducing learning rates during crystallization
    crystal_thresh=4.5e-05,                 # Threshold for identifying weights to crystallize
    
    # Structural plasticity parameters
    rejuvenation_parameter_dre=8.0,         # Controls the rate of weight rejuvenation
    
    # Multi-synaptic connectivity parameters
    weight_splitting_Gamma=2,               # Number of sub-synapses per connection
    weight_splitting_activation_function=nn.ReLU(),  # Activation function for weight splitting
    
    # Volume-dependent plasticity parameters
    base_lr=0.1,                            # Base learning rate for volume-dependent plasticity
    stability_factor=2.0,                   # Controls how quickly stability increases with weight size
    lr_variability=0.2                      # Controls the amount of variability in learning rates
)

converter = BioConverter(config=config)
```

### Distribution Strategies for Fuzzy Learning Rates

Bio Transformations supports various distribution strategies for fuzzy learning rates:

```python
from bio_transformations.bio_config import Distribution

# Different distribution strategies
basic_config = BioConfig(fuzzy_lr_distribution=Distribution.BASELINE)  # All parameters = 1.0 (no variability)
uniform_config = BioConfig(fuzzy_lr_distribution=Distribution.UNIFORM)  # Uniform distribution around 1.0
normal_config = BioConfig(fuzzy_lr_distribution=Distribution.NORMAL)  # Normal distribution centered at 1.0
lognormal_config = BioConfig(fuzzy_lr_distribution=Distribution.LOGNORMAL)  # Log-normal with mean 1.0
gamma_config = BioConfig(fuzzy_lr_distribution=Distribution.GAMMA)  # Gamma distribution (positive, skewed)
beta_config = BioConfig(fuzzy_lr_distribution=Distribution.BETA)  # Beta distribution scaled
layer_config = BioConfig(fuzzy_lr_distribution=Distribution.LAYER_ADAPTIVE)  # Layer-dependent variability
weight_config = BioConfig(fuzzy_lr_distribution=Distribution.WEIGHT_ADAPTIVE)  # Weight-dependent scaling
temporal_config = BioConfig(fuzzy_lr_distribution=Distribution.TEMPORAL, fuzzy_lr_dynamic=True)  # Evolves over time
activity_config = BioConfig(fuzzy_lr_distribution=Distribution.ACTIVITY, fuzzy_lr_dynamic=True)  # Based on activation
```

### Updating Configuration After Creation

You can update the configuration of a `BioConverter` after it has been created:

```python
converter = BioConverter()
converter.update_config(
    dampening_factor=0.7,
    crystal_thresh=5e-05
)

# Or create a converter from a dictionary
config_dict = {
    'fuzzy_learning_rate_factor_nu': 0.2,
    'dampening_factor': 0.7
}
converter = BioConverter.from_dict(config_dict)
```

### Applying Bio Transformations to Existing Models

You can convert existing model instances:

```python
pretrained_model = torchvision.models.resnet18(pretrained=True)
bio_model = converter.convert(pretrained_model)
```

Or use the converter as a decorator:

```python
@converter
class BioResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        # Additional layers...
```

## Extending Functionality

You can extend Bio Transformations with your own bio-inspired methods:

### Adding a New Function to BioModule

1. **Add the Function to BioModule**

```python
# In bio_transformations/bio_module.py
class BioModule(nn.Module):
    # Add your function to the exposed_functions list
    exposed_functions = (
        "rejuvenate_weights",
        "crystallize",
        "fuzzy_learning_rates",
        "volume_dependent_lr",
        "my_new_function",  # <-- Add your function name here
        # ... other existing functions
    )
    
    # Add your function implementation
    def my_new_function(self) -> None:
        """
        Your new bio-inspired function.
        
        This function implements a new bio-inspired mechanism for neural networks.
        """
        # Implementation goes here
        with torch.no_grad():
            # Example: Add random noise to weights
            noise = torch.randn_like(self.get_parent().weight.data) * 0.01
            self.get_parent().weight.data += noise
```

2. **Add parameters to BioConfig if needed**

```python
# In bio_transformations/bio_config.py
class BioConfig(NamedTuple):
    # Existing parameters...
    my_new_parameter: float = 0.5  # Default value for your new parameter
```

3. **Create a test case in test_biomodule.py**

```python
# In test_biomodule.py
def test_my_new_function():
    """Test the my_new_function method of BioModule."""
    linear_layer = nn.Linear(10, 10)
    bio_mod = BioModule(lambda: linear_layer)
    
    # Save initial weights for comparison
    initial_weights = linear_layer.weight.data.clone()
    
    # Call your new function
    bio_mod.my_new_function()
    
    # Verify the function had the expected effect
    assert not torch.allclose(linear_layer.weight.data, initial_weights), "Weights should change after calling my_new_function"
```

4. **Update documentation in the appropriate RST files**

```rst
.. method:: my_new_function()

   Your new bio-inspired function.
   
   This function implements a new bio-inspired mechanism for neural networks.
   
   It uses the `my_new_parameter` from the configuration to control behavior.
```

### Creating a Custom BioModule Class

You can also create your own custom BioModule class with specialized functionality:

```python
from bio_transformations.bio_module import BioModule

class CustomBioModule(BioModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization
        
    def custom_bio_method(self):
        # Your custom bio-inspired logic here
        pass

# Update BioModule.exposed_functions to include your new method
CustomBioModule.exposed_functions = BioModule.exposed_functions + ("custom_bio_method",)

# Use CustomBioModule in your BioConverter
class CustomBioConverter(BioConverter):
    def _bio_modulize(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.add_module('bio_mod', CustomBioModule(lambda: module, **self.bio_module_params))

# Use your custom converter
custom_converter = CustomBioConverter()
bio_model = custom_converter(model)
```

## Contributing Guidelines

We welcome contributions to Bio Transformations! Please follow these steps:

1. Fork the repository and create your branch from `main`.
2. Make changes and ensure all tests pass.
3. Add tests for new functionality.
4. Update documentation to reflect changes.
5. Submit a pull request with a clear description of your changes.

Please adhere to the existing code style and include appropriate comments.

## License Information

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Publication

For more detailed information about the project and its underlying research, please refer to our paper: [DOI]