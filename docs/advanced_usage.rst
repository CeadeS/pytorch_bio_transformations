Advanced Usage
==============

This guide covers advanced topics and customization options for Bio Transformations. It's intended for users who are already familiar with the basic usage of the package and want to explore its full capabilities.

Customizing BioConverter with Comprehensive Configuration
--------------------------------------------------------

The ``BioConfig`` class provides extensive configuration options for fine-tuning the bio-inspired modifications. Here's an example of creating a ``BioConverter`` with custom settings across all features:

.. code-block:: python

   from bio_transformations import BioConverter, BioConfig
   from bio_transformations.bio_config import Distribution
   import torch.nn as nn

   # Comprehensive configuration
   converter = BioConverter(
       # Fuzzy learning rate parameters
       fuzzy_learning_rate_factor_nu=0.2,        # Controls the variability in learning rates
       fuzzy_lr_distribution=Distribution.NORMAL, # Distribution strategy for learning rates
       fuzzy_lr_dynamic=True,                     # Whether to update learning rates during training
       fuzzy_lr_min=0.5,                          # Minimum value for learning rates
       fuzzy_lr_max=1.5,                          # Maximum value for learning rates
       fuzzy_lr_update_freq=5,                    # How often to update temporal rates
       fuzzy_lr_decay=0.98,                       # Decay factor for temporal rates

       # Synaptic stabilization parameters
       dampening_factor=0.7,                      # Factor for reducing learning rates during crystallization
       crystal_thresh=5e-05,                      # Threshold for identifying weights to crystallize

       # Structural plasticity parameters
       rejuvenation_parameter_dre=10.0,           # Controls the rate of weight rejuvenation

       # Multi-synaptic connectivity parameters
       weight_splitting_Gamma=2,                  # Number of sub-synapses per connection (0 = disabled)
       weight_splitting_activation_function=nn.ReLU(),  # Activation for weight splitting

       # Volume-dependent plasticity parameters
       base_lr=0.05,                              # Base learning rate for volume-dependent plasticity
       stability_factor=2.5,                      # Controls how quickly stability increases with weight size
       lr_variability=0.15,                       # Controls the amount of variability in learning rates

       # Dale's principle parameters
       apply_dales_principle=True                 # Whether to enforce Dale's principle
   )

   # Convert a model with this comprehensive configuration
   bio_model = converter(model)

Creating and Updating BioConverter from Dictionary
-------------------------------------------------

You can create a ``BioConverter`` from a dictionary of parameters or update an existing converter:

.. code-block:: python

   # Create from dictionary
   config_dict = {
       'fuzzy_learning_rate_factor_nu': 0.2,
       'dampening_factor': 0.7,
       'crystal_thresh': 5e-05,
       'rejuvenation_parameter_dre': 10.0,
       'weight_splitting_Gamma': 2
   }

   converter = BioConverter.from_dict(config_dict)

   # Update an existing converter
   converter.update_config(
       fuzzy_lr_distribution=Distribution.LOGNORMAL,
       fuzzy_lr_dynamic=True,
       apply_dales_principle=True
   )

   # Get the current configuration
   current_config = converter.get_config()
   print(current_config)

Detailed Guide to Fuzzy Learning Rate Distributions
--------------------------------------------------

Bio Transformations offers multiple distribution strategies for fuzzy learning rates, each with unique characteristics:

.. code-block:: python

   from bio_transformations.bio_config import Distribution

   # 1. BASELINE - No variability (all parameters = 1.0)
   baseline_config = BioConfig(fuzzy_lr_distribution=Distribution.BASELINE)
   # Good for establishing a performance baseline without diversity

   # 2. UNIFORM - Uniform distribution around 1.0
   uniform_config = BioConfig(
       fuzzy_lr_distribution=Distribution.UNIFORM,
       fuzzy_learning_rate_factor_nu=0.16  # Controls the range: [1-0.16, 1+0.16]
   )
   # Simple, predictable variability across all weights

   # 3. NORMAL - Normal distribution centered at 1.0
   normal_config = BioConfig(
       fuzzy_lr_distribution=Distribution.NORMAL,
       fuzzy_learning_rate_factor_nu=0.16  # Standard deviation
   )
   # Bell-curve distribution with most values near 1.0

   # 4. LOGNORMAL - Log-normal with mean 1.0
   lognormal_config = BioConfig(
       fuzzy_lr_distribution=Distribution.LOGNORMAL,
       fuzzy_learning_rate_factor_nu=0.16  # Controls the shape
   )
   # Skewed distribution with long tail, all positive values

   # 5. GAMMA - Gamma distribution (positive, skewed)
   gamma_config = BioConfig(
       fuzzy_lr_distribution=Distribution.GAMMA,
       fuzzy_learning_rate_factor_nu=0.16  # Controls the shape
   )
   # Models continuous waiting times, good for activity-dependent processes

   # 6. BETA - Beta distribution scaled to [1-nu, 1+nu]
   beta_config = BioConfig(
       fuzzy_lr_distribution=Distribution.BETA,
       fuzzy_learning_rate_factor_nu=0.16  # Controls the shape and range
   )
   # Flexible distribution bounded on both sides

   # 7. LAYER_ADAPTIVE - Layer-dependent variability
   layer_config = BioConfig(
       fuzzy_lr_distribution=Distribution.LAYER_ADAPTIVE,
       fuzzy_learning_rate_factor_nu=0.16  # Base variability
   )
   # Early layers get more variability than later layers
   # Mimics biological observation of layer-specific plasticity in cortex

   # 8. WEIGHT_ADAPTIVE - Weight-dependent scaling
   weight_config = BioConfig(
       fuzzy_lr_distribution=Distribution.WEIGHT_ADAPTIVE,
       fuzzy_learning_rate_factor_nu=0.16  # Base variability
   )
   # Smaller weights get more variability than larger weights
   # Mimics size-dependent plasticity of dendritic spines

   # 9. TEMPORAL - Evolves over time
   temporal_config = BioConfig(
       fuzzy_lr_distribution=Distribution.TEMPORAL,
       fuzzy_lr_dynamic=True,               # Must be True for temporal evolution
       fuzzy_learning_rate_factor_nu=0.16,  # Base variability
       fuzzy_lr_update_freq=10,             # Update every 10 steps
       fuzzy_lr_decay=0.95                  # Decay factor for temporal rates
   )
   # Learning rates change during training, mimicking developmental changes

   # 10. ACTIVITY - Based on neuron activation patterns
   activity_config = BioConfig(
       fuzzy_lr_distribution=Distribution.ACTIVITY,
       fuzzy_lr_dynamic=True,               # Must be True for activity tracking
       fuzzy_learning_rate_factor_nu=0.16   # Base variability
   )
   # Adjusts learning rates based on neuron activity
   # More active neurons become more stable (less variable)

Implementing Custom Activation Functions for Weight Splitting
-------------------------------------------------------------

You can implement custom activation functions for weight splitting to modify how multi-synaptic connections behave:

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F

   # Custom activation: Leaky ReLU with specific negative slope
   def custom_leaky_activation(x):
       return F.leaky_relu(x, negative_slope=0.05)

   # Custom activation: Sigmoid with scaling
   def custom_sigmoid_activation(x):
       return torch.sigmoid(x) * 2.0  # Scale output range to [0, 2]

   # Custom activation: Tanh with gain
   def custom_tanh_activation(x):
       return torch.tanh(x * 1.5)  # Apply gain before tanh

   # Use the custom activation in BioConverter
   converter = BioConverter(
       weight_splitting_Gamma=2,  # Enable weight splitting
       weight_splitting_activation_function=custom_leaky_activation
   )

   bio_model = converter(model)

Selective Application of Bio-Inspired Features
----------------------------------------------

You can selectively apply bio-inspired features to specific layers of your model:

.. code-block:: python

   class CustomModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(10, 20)
           self.fc2 = nn.Linear(20, 5)
           self.fc3 = nn.Linear(5, 1)

       def forward(self, x):
           x = torch.relu(self.fc1(x))
           x = torch.relu(self.fc2(x))
           return self.fc3(x)

   model = CustomModel()

   # Mark fc3 to skip weight splitting
   # This is particularly useful for output layers where
   # changing the output dimension would affect the task
   BioConverter.set_last_module_token_for_module(model.fc3)

   # Convert the model
   bio_model = converter(model)

Monitoring Bio-Inspired Modifications
-------------------------------------

To monitor the effects of bio-inspired modifications during training:

.. code-block:: python

   class MonitoredModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(10, 20)
           self.fc2 = nn.Linear(20, 5)

       def forward(self, x):
           x = torch.relu(self.fc1(x))
           return self.fc2(x)

   model = MonitoredModel()
   converter = BioConverter()
   bio_model = converter(model)

   # Track changes during training
   crystallized_weights_history = []
   rejuvenated_weights_history = []
   learning_rate_variability_history = []

   # Training loop with monitoring
   for epoch in range(100):
       # Forward and backward pass
       outputs = bio_model(inputs)
       loss = criterion(outputs, targets)
       optimizer.zero_grad()
       loss.backward()

       # Get pre-modification weights for comparison
       pre_weights_fc1 = bio_model.fc1.weight.data.clone()

       # Apply bio-inspired modifications
       bio_model.crystallize()

       # Track crystallized weights after each epoch
       with torch.no_grad():
           # Count crystallized weights (those with reduced learning rates)
           crystallized_count = torch.sum(
               bio_model.fc1.bio_mod.fuzzy_learning_rate_parameters < 0.9
           ).item()
           crystallized_weights_history.append(crystallized_count)

           # Check learning rate variability
           lr_variability = bio_model.fc1.bio_mod.fuzzy_learning_rate_parameters.std().item()
           learning_rate_variability_history.append(lr_variability)

       # Apply other modifications
       bio_model.fuzzy_learning_rates()
       optimizer.step()

       # Periodically apply rejuvenation
       if epoch % 10 == 0:
           # Get pre-rejuvenation weights
           pre_rejuv_weights = bio_model.fc1.weight.data.clone()

           # Apply rejuvenation
           bio_model.rejuvenate_weights()

           # Count rejuvenated weights
           with torch.no_grad():
               rejuvenated_count = torch.sum(
                   (bio_model.fc1.weight.data - pre_rejuv_weights).abs() > 1e-6
               ).item()
               rejuvenated_weights_history.append(rejuvenated_count)

   # Plot the results
   import matplotlib.pyplot as plt

   plt.figure(figsize=(15, 5))

   plt.subplot(1, 3, 1)
   plt.plot(crystallized_weights_history)
   plt.title('Crystallized Weights Over Time')
   plt.xlabel('Epoch')
   plt.ylabel('Number of Crystallized Weights')

   plt.subplot(1, 3, 2)
   plt.plot(rejuvenated_weights_history)
   plt.title('Rejuvenated Weights Over Time')
   plt.xlabel('Epoch')
   plt.ylabel('Number of Rejuvenated Weights')

   plt.subplot(1, 3, 3)
   plt.plot(learning_rate_variability_history)
   plt.title('Learning Rate Variability Over Time')
   plt.xlabel('Epoch')
   plt.ylabel('Standard Deviation of Learning Rates')

   plt.tight_layout()
   plt.show()

Creating a Custom BioModule Extension
-------------------------------------

You can extend the ``BioModule`` class with your own bio-inspired methods:

.. code-block:: python

   from bio_transformations.bio_module import BioModule
   from bio_transformations import BioConverter

   class CustomBioModule(BioModule):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           # Add custom state variables
           self.register_buffer('activity_history', torch.zeros(10))
           self.current_step = 0

       def custom_bio_method(self):
           """
           Custom bio-inspired plasticity rule that scales weights
           based on a simulated neuromodulator presence.
           """
           with torch.no_grad():
               # Simulate neuromodulator concentration varying over time
               neuromodulator = 0.5 + 0.5 * torch.sin(torch.tensor(self.current_step / 10))
               self.current_step += 1

               # Scale weights based on neuromodulator concentration
               scale_factor = 1.0 + 0.1 * neuromodulator
               self.get_parent().weight.data *= scale_factor

               # Track activity for this step
               idx = self.current_step % 10
               self.activity_history[idx] = neuromodulator.item()

   # Update BioModule.exposed_functions to include your new method
   CustomBioModule.exposed_functions = BioModule.exposed_functions + ("custom_bio_method",)

   # Create a custom BioConverter that uses your extended BioModule
   class CustomBioConverter(BioConverter):
       def _bio_modulize(self, module):
           if isinstance(module, (nn.Linear, nn.Conv2d)):
               module.add_module('bio_mod', CustomBioModule(lambda: module, config=self.config))

   # Use your custom converter
   custom_converter = CustomBioConverter()
   bio_model = custom_converter(model)

   # Training loop with custom method
   for epoch in range(100):
       # Standard training steps
       # ...

       # Apply custom bio method
       bio_model.custom_bio_method()

       # Continue with other modifications
       # ...

Combining Bio Transformations with Other PyTorch Features
---------------------------------------------------------

Bio Transformations can be combined with other PyTorch features like DataParallel for multi-GPU training or TorchScript for deployment:

.. code-block:: python

   import torch.nn as nn
   from torch.nn.parallel import DataParallel
   from bio_transformations import BioConverter

   # Create and convert model
   model = YourModel()
   converter = BioConverter()
   bio_model = converter(model)

   # Wrap with DataParallel for multi-GPU training
   if torch.cuda.device_count() > 1:
       bio_model = DataParallel(bio_model)

   bio_model = bio_model.to(device)

   # Standard training loop
   for inputs, targets in train_loader:
       inputs, targets = inputs.to(device), targets.to(device)
       outputs = bio_model(inputs)
       loss = criterion(outputs, targets)
       optimizer.zero_grad()
       loss.backward()

       # For DataParallel models, we need to access the module attribute
       if isinstance(bio_model, DataParallel):
           bio_model.module.volume_dependent_lr()
           bio_model.module.fuzzy_learning_rates()
           bio_model.module.crystallize()
       else:
           bio_model.volume_dependent_lr()
           bio_model.fuzzy_learning_rates()
           bio_model.crystallize()

       optimizer.step()

   # Export with TorchScript
   # Note: After conversion to TorchScript, bio-inspired methods
   # can no longer be called, so this is for deployment only
   scripted_model = torch.jit.script(bio_model)
   scripted_model.save("bio_model_scripted.pt")

Performance Optimization Tips
--------------------------

Here are some tips to optimize performance when using Bio Transformations:

1. **Selective Application of Bio-Inspired Methods**

   Not all bio-inspired methods need to be applied at every iteration. For example:

   .. code-block:: python

      # Instead of applying everything every iteration:
      for i, (inputs, targets) in enumerate(train_loader):
          outputs = bio_model(inputs)
          loss = criterion(outputs, targets)
          optimizer.zero_grad()
          loss.backward()

          bio_model.volume_dependent_lr()
          bio_model.fuzzy_learning_rates()
          bio_model.crystallize()
          bio_model.rejuvenate_weights()  # Expensive operation

          optimizer.step()

      # Consider selective application:
      for i, (inputs, targets) in enumerate(train_loader):
          outputs = bio_model(inputs)
          loss = criterion(outputs, targets)
          optimizer.zero_grad()
          loss.backward()

          # Apply these every iteration
          bio_model.fuzzy_learning_rates()

          # Apply some methods less frequently
          if i % 10 == 0:
              bio_model.crystallize()

          # Apply expensive operations very selectively
          if i % 100 == 0:
              bio_model.rejuvenate_weights()

          optimizer.step()

2. **Use Non-Dynamic Distributions When Possible**

   Dynamic distributions require updates during training and may be more computationally expensive:

   .. code-block:: python

      # More efficient (no updates required during training):
      config = BioConfig(
          fuzzy_lr_distribution=Distribution.NORMAL,
          fuzzy_lr_dynamic=False
      )

      # Less efficient (requires updates during training):
      config = BioConfig(
          fuzzy_lr_distribution=Distribution.TEMPORAL,
          fuzzy_lr_dynamic=True
      )

3. **Batch Processing for Activity-Dependent Learning**

   When using activity-dependent learning rates, process in batches rather than single examples:

   .. code-block:: python

      # Process entire batches (more efficient):
      outputs = bio_model(inputs_batch)  # inputs_batch shape: [batch_size, features]
      bio_model.update_fuzzy_learning_rates(inputs_batch)

      # Avoid processing individual examples (less efficient):
      for single_input in inputs_batch:
          output = bio_model(single_input.unsqueeze(0))
          bio_model.update_fuzzy_learning_rates(single_input.unsqueeze(0))

4. **Consider Model Size vs. Weight Splitting**

   Weight splitting increases memory usage and computation. For large models, consider using smaller values:

   .. code-block:: python

      # For smaller models, more splitting might be fine:
      small_model_converter = BioConverter(weight_splitting_Gamma=4)

      # For larger models, use less splitting or disable it:
      large_model_converter = BioConverter(weight_splitting_Gamma=2)  # Less splitting
      very_large_model_converter = BioConverter(weight_splitting_Gamma=0)  # Disabled

5. **Adjust Bio-Inspired Parameters Based on Network Size**

   Larger networks may require different parameter settings:

   .. code-block:: python

      # For small networks:
      small_config = BioConfig(
          fuzzy_learning_rate_factor_nu=0.16,
          rejuvenation_parameter_dre=8.0
      )

      # For large networks, use more conservative settings:
      large_config = BioConfig(
          fuzzy_learning_rate_factor_nu=0.1,  # Less variability
          rejuvenation_parameter_dre=12.0     # Less aggressive rejuvenation
      )

These advanced usage examples should help you customize and extend Bio Transformations to suit your specific needs. Remember to refer to the API documentation for detailed information on each class and method.