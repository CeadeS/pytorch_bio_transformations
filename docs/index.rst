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
