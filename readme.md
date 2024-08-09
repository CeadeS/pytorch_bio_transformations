[![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/mit)
![PyPI](https://img.shields.io/pypi/v/pytorch_bio_transformations)
[![tests](https://github.com/CeadeS/pytorch_bio_transformations/actions/workflows/test.yml/badge.svg)](https://github.com/CeadeS/pytorch_bio_transformations/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/CeadeS/pytorch_bio_transformations/branch/dev/graph/badge.svg?token=I11PUI5K0S)](https://codecov.io/gh/CeadeS/pytorch_bio_transformations)
[![Perform tests](https://github.com/CeadeS/pytorch_bio_transformations/actions/workflows/test.yml/badge.svg)](https://github.com/CeadeS/pytorch_bio_transformations/actions/workflows/test.yml)
[![Build and deploy docs](https://github.com/CeadeS/pytorch_bio_transformations/actions/workflows/documentation.yml/badge.svg)](https://github.com/CeadeS/pytorch_bio_transformations/actions/workflows/documentation.yml)
[![Release](https://github.com/CeadeS/pytorch_bio_transformations/actions/workflows/release_and_deploy.yml/badge.svg)](https://github.com/CeadeS/pytorch_bio_transformations/actions/workflows/release_and_deploy.yml)

# BioLearn: Biologically Inspired Neural Network Modifications

Please visit the [Documentation](https://ceades.github.io/pytorch_bio_transformations/index.html) for further information or refer to the [Publication](#publication)

## Table of Contents
1. [Project Description](#project-description)
2. [Key Features](#key-features)
3. [Installation Instructions](#installation-instructions)
4. [Usage](#usage)
5. [Contributing Guidelines](#contributing-guidelines)
6. [License Information](#license-information)
7. [Publication](#publication)

## Project Description

BioLearn is a Python library that implements biologically inspired modifications to artificial neural networks, based on research on dendritic spine dynamics. It aims to explore and enhance the learning capabilities of neural networks by mimicking the plasticity and stability characteristics observed in biological synapses.

This project is primarily targeted at researchers and developers in the fields of machine learning and computational neuroscience who are interested in exploring bio-inspired approaches to augment neural network performance.

## Key Features

BioLearn implements several biologically inspired methods, each mimicking specific aspects of neuronal behavior:

1. `rejuvenate_weights`: Simulates spine turnover, replacing weak synapses with new ones.
2. `crystallize`: Mimics synaptic stabilization, adjusting learning rates based on synaptic strength and activity.
3. `fuzzy_learning_rates`: Implements synaptic scaling for network stability.
4. `weight_splitting`: Replicates multi-synaptic connectivity between neuron pairs.
5. `volume_dependent_lr`: Applies learning rates based on synaptic "volume", inspired by spine size-plasticity relationships.

These methods work in concert to create a learning process that more closely resembles the dynamics observed in biological neural networks, potentially leading to improved learning and generalization in artificial neural networks.

## Installation Instructions

You can install BioLearn using pip, Conda, or from source.

### Option 1: Using pip (Simplest Method)

```bash
pip install bio_transformations
```

### Option 2: Using Conda

```bash
conda create -n biolearn python=3.8
conda activate biolearn
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install bio_transformations
```

### Option 3: From Source (For Development or Latest Changes)

```bash
git clone https://github.com/CeadeS/pytorch_bio_transformations
cd pytorch_bio_transformations
pip install -r requirements.txt
pip install -e .
```

### Verifying Installation

```bash
python -c "import bio_transformations; print(bio_transformations.__version__)"
```

## Usage

### Basic Usage Example

```python
import torch
import torch.nn as nn
from bio_transformations import BioConverter

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleModel()
converter = BioConverter(base_lr=0.1, stability_factor=2.0, lr_variability=0.2)
bio_model = converter(model)
```

### Training Example

```python
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(bio_model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch in data_loader:
        inputs, targets = batch
        outputs = bio_model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        
        bio_model.volume_dependent_lr()
        bio_model.crystallize()
        
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

### Adding Your Own Function

To add a new function to BioModule:

1. Add the function to the `BioModule` class in `bio_module.py`.
2. Add the function name to the `exposed_functions` list in `BioModule`.
3. Update the `BioConverter` class in `bio_converter.py` if needed.
4. Create a test case in `test_biomodule.py`.

## Contributing Guidelines

We welcome contributions to BioLearn! Please follow these steps:

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
