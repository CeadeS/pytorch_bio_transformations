.. Bio Transformations documentation master file, created by
   sphinx-quickstart on Thu Aug  8 09:19:09 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/CeadeS/pytorch_bio_transformations/

Welcome to Bio Transformations
==============================

Bio Transformations is a Python package that enhances artificial neural networks (ANNs) by incorporating biologically inspired mechanisms observed in biological neural networks (BNNs). Our goal is to improve the learning speed, prediction accuracy, and resilience of ANNs using concepts from neuroscience.

Quick Start
-----------

Installation
^^^^^^^^^^^^

Install Bio Transformations using pip:

.. code-block:: bash

   pip install pytorch_bio_transformations

Basic Usage
^^^^^^^^^^^

Here's a simple example to get you started:

.. code-block:: python

   import torch.nn as nn
   from bio_transformations import BioConverter

   # Define your model
   class SimpleModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(10, 20)
           self.fc2 = nn.Linear(20, 1)

       def forward(self, x):
           x = nn.functional.relu(self.fc1(x))
           return self.fc2(x)

   # Create and convert your model
   model = SimpleModel()
   converter = BioConverter(
       fuzzy_learning_rate_factor_nu=0.16,
       dampening_factor=0.6,
       crystal_thresh=4.5e-05
   )
   bio_model = converter(model)

   # Use bio_model as you would a regular PyTorch model

Configuring Fuzzy Learning Rates
^^^^^^^^^^^

.. code-block:: python

   import torch
   import torch.nn as nn
   from bio_transformations import BioConverter
   from bio_transformations.bio_config import BioConfig, Distribution

   # Create a model
   model = nn.Sequential(
       nn.Linear(10, 20),
       nn.ReLU(),
       nn.Linear(20, 5)
   )

   # Configure with log-normal distribution
   config = BioConfig(
       fuzzy_learning_rate_factor_nu=0.2,
       fuzzy_lr_distribution=Distribution.LOGNORMAL,
       fuzzy_lr_min=0.7,
       fuzzy_lr_max=1.5
   )
   converter = BioConverter(config=config)
   bio_model = converter(model)

   # Or use the weight-adaptive approach
   config = BioConfig(
       fuzzy_learning_rate_factor_nu=0.16,
       fuzzy_lr_distribution=Distribution.WEIGHT_ADAPTIVE,
   )
   converter = BioConverter(config=config)
   bio_model = converter(model)

   # Or try the dynamic temporal approach
   config = BioConfig(
       fuzzy_learning_rate_factor_nu=0.16,
       fuzzy_lr_distribution=Distribution.TEMPORAL,
       fuzzy_lr_dynamic=True,
       fuzzy_lr_update_freq=50  # Update every 50 steps
   )
   converter = BioConverter(config=config)
   bio_model = converter(model)

   # Training loop with dynamic updates
   for epoch in range(100):
       # Forward pass
       output = bio_model(inputs)
       loss = criterion(output, targets)

       # Backward pass
       optimizer.zero_grad()
       loss.backward()

       # Apply fuzzy learning rates
       bio_model.fuzzy_learning_rates()

       # Update fuzzy learning rates if using dynamic strategies
       bio_model.update_fuzzy_learning_rates()

       optimizer.step()


Key Concepts
------------

Bio Transformations implements three key biologically inspired mechanisms:

1. **Diversity in synaptic plasticity**: Not all synapses learn at the same rate.
2. **Spontaneous spine remodeling**: Synapses can form and disappear dynamically.
3. **Multi-synaptic connectivity**: Multiple connections can exist between neuron pairs.

These concepts are implemented through various methods such as `fuzzy_learning_rates()`, `rejuvenate_weights()`, and `add_weight_splitting_step()`.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   concepts
   tutorials
   advanced_usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules

About the Project
-----------------

Bio Transformations is based on the paper "Synaptic Diversity: Concept Transfer from Biological to Artificial Neural Networks" by Martin Hofmann, Moritz Franz Peter Becker, Christian Tetzlaff, and Patrick Mäder. Our package aims to bridge the gap between biological and artificial neural networks, potentially leading to more efficient and robust AI systems.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
