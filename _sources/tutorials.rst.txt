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

   # Create a BioConverter with default parameters
   converter = BioConverter()

   # Convert the model
   bio_model = converter(model)

   # Now bio_model can be used like a regular PyTorch model
   x = torch.randn(1, 10)
   output = bio_model(x)
   print(output)

Customizing the Bio-inspired Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can customize the bio-inspired features by passing parameters to the BioConverter.

.. code-block:: python

   # Create a BioConverter with custom parameters
   converter = BioConverter(
       fuzzy_learning_rate_factor_nu=0.16,  # Controls the diversity in learning rates
       dampening_factor=0.6,                # Controls the stability increase during crystallization
       crystal_thresh=4.5e-05,              # Threshold for identifying weights to crystallize
       rejuvenation_parameter_dre=8.0,      # Controls the rate of weight rejuvenation
       weight_splitting_Gamma=2,            # Number of sub-synapses per connection (0 = disabled)
       apply_dales_principle=False          # Whether to enforce Dale's principle
   )

   # Convert the model with custom parameters
   bio_model = converter(model)

Training a Bio-Transformed Model
--------------------------------

Here's how to train a model using Bio Transformations:

.. code-block:: python

   import torch.optim as optim

   # Assume we have our bio_model from the previous example

   # Define loss function and optimizer
   criterion = nn.MSELoss()
   optimizer = optim.Adam(bio_model.parameters(), lr=0.01)

   # Training data (example)
   x_train = torch.randn(100, 10)
   y_train = torch.randn(100, 1)

   # Training loop
   for epoch in range(100):  # 100 epochs
       # Forward pass
       optimizer.zero_grad()
       outputs = bio_model(x_train)
       loss = criterion(outputs, y_train)
       loss.backward()

       # Apply bio-inspired modifications
       bio_model.volume_dependent_lr()   # Adjust learning rates based on weight size
       bio_model.fuzzy_learning_rates()  # Apply diverse learning rates
       bio_model.crystallize()           # Stabilize well-optimized weights

       # Update weights
       optimizer.step()

       # Periodically apply weight rejuvenation (e.g., every 10 epochs)
       if epoch % 10 == 0:
           bio_model.rejuvenate_weights()

       if epoch % 10 == 0:
           print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

Using Different Distribution Strategies
--------------------------------------

Bio Transformations supports various distribution strategies for fuzzy learning rates:

.. code-block:: python

   from bio_transformations import BioConverter, BioConfig
   from bio_transformations.bio_config import Distribution

   # Create a model
   model = SimpleModel()

   # Create a BioConverter with Normal distribution
   normal_config = BioConfig(
       fuzzy_lr_distribution=Distribution.NORMAL,
       fuzzy_learning_rate_factor_nu=0.16  # Standard deviation for normal distribution
   )
   converter = BioConverter(config=normal_config)
   bio_model = converter(model)

   # Train with normal distribution (just like previous example)
   # ...

   # Create a BioConverter with Weight-adaptive distribution
   weight_config = BioConfig(
       fuzzy_lr_distribution=Distribution.WEIGHT_ADAPTIVE,
       fuzzy_learning_rate_factor_nu=0.16
   )
   converter = BioConverter(config=weight_config)
   bio_model = converter(model)

   # Train with weight-adaptive distribution
   # (smaller weights get more variability than larger weights)
   # ...

Using Activity-Dependent Learning Rates
--------------------------------------

For activity-dependent learning rates, you need to pass the input tensor to the update_fuzzy_learning_rates method:

.. code-block:: python

   from bio_transformations import BioConverter, BioConfig
   from bio_transformations.bio_config import Distribution

   # Create a model
   model = SimpleModel()

   # Create a BioConverter with Activity-dependent distribution
   activity_config = BioConfig(
       fuzzy_lr_distribution=Distribution.ACTIVITY,
       fuzzy_lr_dynamic=True,  # Important: must be True for activity-dependent rates
       fuzzy_learning_rate_factor_nu=0.16
   )
   converter = BioConverter(config=activity_config)
   bio_model = converter(model)

   # Training loop
   for epoch in range(100):
       # Forward pass
       optimizer.zero_grad()
       outputs = bio_model(x_train)  # Pass input through the model
       loss = criterion(outputs, y_train)
       loss.backward()

       # Update fuzzy learning rates based on activity
       # This reads the activation patterns from the last forward pass
       bio_model.update_fuzzy_learning_rates(x_train)

       # Apply other bio-inspired modifications
       bio_model.fuzzy_learning_rates()
       optimizer.step()

Applying Dale's Principle
------------------------

To enforce Dale's principle (neurons are either excitatory or inhibitory):

.. code-block:: python

   from bio_transformations import BioConverter, BioConfig

   # Create a model
   model = SimpleModel()

   # Create a BioConverter with Dale's principle enabled
   dales_config = BioConfig(
       apply_dales_principle=True
   )
   converter = BioConverter(config=dales_config)
   bio_model = converter(model)

   # Training loop
   for epoch in range(100):
       # Standard training steps
       # ...

       # Enforce Dale's principle after each update
       bio_model.enforce_dales_principle()

       # Continue with other modifications
       # ...

Using Bio Transformations with CNNs
----------------------------------

Bio Transformations works with convolutional neural networks too:

.. code-block:: python

   import torch.nn as nn
   from bio_transformations import BioConverter

   # Define a simple CNN
   class SimpleCNN(nn.Module):
       def __init__(self):
           super().__init__()
           self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
           self.pool = nn.MaxPool2d(2, 2)
           self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
           self.fc1 = nn.Linear(32 * 7 * 7, 128)
           self.fc2 = nn.Linear(128, 10)

       def forward(self, x):
           x = self.pool(torch.relu(self.conv1(x)))
           x = self.pool(torch.relu(self.conv2(x)))
           x = x.view(-1, 32 * 7 * 7)
           x = torch.relu(self.fc1(x))
           return self.fc2(x)

   # Create and convert the CNN
   cnn_model = SimpleCNN()
   converter = BioConverter()
   bio_cnn = converter(cnn_model)

   # Use bio_cnn just like a regular CNN

Troubleshooting Common Issues
----------------------------------

Here are solutions to common issues you might encounter when using Bio Transformations:

RuntimeError: "No gradients found for the weights"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This error occurs when calling ``fuzzy_learning_rates()``, ``volume_dependent_lr()``, or ``crystallize()`` before performing a backward pass:

.. code-block:: python

   # Problem:
   outputs = bio_model(inputs)
   bio_model.fuzzy_learning_rates()  # Error! No gradients yet

   # Solution:
   outputs = bio_model(inputs)
   loss = criterion(outputs, targets)
   optimizer.zero_grad()
   loss.backward()  # This generates gradients
   bio_model.fuzzy_learning_rates()  # Now works correctly

ValueError: "weight_splitting_Gamma must evenly divide the number of features"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This occurs when the number of features in a layer is not divisible by ``weight_splitting_Gamma``:

.. code-block:: python

   # Problem:
   model = nn.Linear(10, 15)  # 15 is not divisible by weight_splitting_Gamma=2
   converter = BioConverter(weight_splitting_Gamma=2)
   bio_model = converter(model)  # Error!

   # Solution 1: Choose a compatible weight_splitting_Gamma
   converter = BioConverter(weight_splitting_Gamma=3)  # 15 is divisible by 3

   # Solution 2: Adjust your model architecture
   model = nn.Linear(10, 16)  # 16 is divisible by 2
   converter = BioConverter(weight_splitting_Gamma=2)

AttributeError: "Can not enforce dales principle without apply_dales_principle set True"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This occurs when trying to call ``enforce_dales_principle()`` but Dale's principle is not enabled:

.. code-block:: python

   # Problem:
   converter = BioConverter()  # apply_dales_principle defaults to False
   bio_model = converter(model)
   bio_model.enforce_dales_principle()  # Error!

   # Solution:
   converter = BioConverter(apply_dales_principle=True)
   bio_model = converter(model)
   bio_model.enforce_dales_principle()  # Now works correctly

These tutorials should help you get started with Bio Transformations. For more advanced usage and customization options, please refer to the Advanced Usage guide.