Tutorials
=========

This section provides step-by-step tutorials to help you get started with Bio Transformations. We'll cover basic usage, integration with existing models, and how to use specific bio-inspired features.

Basic Usage
-----------

Converting a Simple Model
^^^^^^^^^^^^^^^^^^^^^^^^^

In this tutorial, we'll convert a simple PyTorch model to use Bio Transformations.

.. code-block:: python

   import torch
   import torch.nn as nn
   from bio_transformations import BioConverter

   # Define a simple model
   class SimpleModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(10, 20)
           self.fc2 = nn.Linear(20, 1)

       def forward(self, x):
           x = torch.relu(self.fc1(x))
           return self.fc2(x)

   # Create an instance of the model
   model = SimpleModel()

   # Create a BioConverter
   converter = BioConverter(
       fuzzy_learning_rate_factor_nu=0.16,
       dampening_factor=0.6,
       crystal_thresh=4.5e-05
   )

   # Convert the model
   bio_model = converter(model)

   # Now bio_model can be used like a regular PyTorch model
   x = torch.randn(1, 10)
   output = bio_model(x)
   print(output)

Training a Bio-Transformed Model
--------------------------------

Here's how to train a model using Bio Transformations:

.. code-block:: python

   import torch.optim as optim

   # Assume we have our bio_model from the previous example

   # Define loss function and optimizer
   criterion = nn.MSELoss()
   optimizer = optim.Adam(bio_model.parameters(), lr=0.01)

   # Training loop
   for epoch in range(100):  # 100 epochs
       # Assume we have some training data x_train, y_train
       optimizer.zero_grad()
       outputs = bio_model(x_train)
       loss = criterion(outputs, y_train)
       loss.backward()

       # Apply bio-inspired modifications
       bio_model.volume_dependent_lr()
       bio_model.crystallize()

       optimizer.step()

       if epoch % 10 == 0:
           print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

Using Specific Bio-Inspired Features
------------------------------------

Applying Fuzzy Learning Rates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use fuzzy learning rates:

.. code-block:: python

   # During training
   optimizer.zero_grad()
   outputs = bio_model(x_train)
   loss = criterion(outputs, y_train)
   loss.backward()

   # Apply fuzzy learning rates
   bio_model.fuzzy_learning_rates()

   optimizer.step()

Rejuvenating Weights
^^^^^^^^^^^^^^^^^^^^

To rejuvenate weights:

.. code-block:: python

   # Periodically during training, e.g., every 10 epochs
   if epoch % 10 == 0:
       bio_model.rejuvenate_weights()

These tutorials should help you get started with Bio Transformations. For more advanced usage and customization options, please refer to the Advanced Usage guide.