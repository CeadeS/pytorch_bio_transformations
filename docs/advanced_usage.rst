Advanced Usage
==============

This guide covers advanced topics and customization options for Bio Transformations. It's intended for users who are already familiar with the basic usage of the package and want to explore its full capabilities.

Customizing BioConverter
------------------------

The `BioConverter` class accepts several parameters that allow you to fine-tune the bio-inspired modifications. Here's an example of creating a `BioConverter` with custom settings:

.. code-block:: python

   from bio_transformations import BioConverter

   converter = BioConverter(
       fuzzy_learning_rate_factor_nu=0.2,
       dampening_factor=0.7,
       crystal_thresh=5e-05,
       rejuvenation_parameter_dre=10.0,
       weight_splitting_Gamma=3,
       apply_dales_principle=True,
       base_lr=0.05,
       stability_factor=2.5,
       lr_variability=0.15
   )

Refer to the API documentation for detailed explanations of each parameter.

Implementing Custom Activation Functions
----------------------------------------

You can implement custom activation functions for weight splitting. Here's an example:

.. code-block:: python

   import torch

   def custom_activation(x):
       return torch.tanh(x) * x

   converter = BioConverter(
       weight_splitting_activation_function=custom_activation,
       weight_splitting_Gamma=2
   )

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
   BioConverter.mark_skip_weight_splitting(model.fc3)

   # Convert the model
   bio_model = converter(model)

Extending BioModule with Custom Methods
---------------------------------------

You can extend the `BioModule` class with your own bio-inspired methods:

.. code-block:: python

   from bio_transformations.bio_module import BioModule

   class CustomBioModule(BioModule):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)

       def custom_bio_method(self):
           # Your custom bio-inspired logic here
           pass

   # Update BioModule.exposed_functions to include your new method
   BioModule.exposed_functions += ("custom_bio_method",)

   # Use CustomBioModule in your BioConverter
   class CustomBioConverter(BioConverter):
       def _bio_modulize(self, module):
           if isinstance(module, (nn.Linear, nn.Conv2d)):
               module.add_module('bio_mod', CustomBioModule(lambda: module, **self.bio_module_params))

   # Use your custom converter
   custom_converter = CustomBioConverter()
   bio_model = custom_converter(model)

   # Now you can use your custom method
   bio_model.custom_bio_method()

Monitoring Bio-Inspired Modifications
-------------------------------------

To monitor the effects of bio-inspired modifications during training:

.. code-block:: python

   class MonitoredBioModule(BioModule):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.weight_changes = []

       def rejuvenate_weights(self):
           before = self.get_parent().weight.data.clone()
           super().rejuvenate_weights()
           after = self.get_parent().weight.data
           self.weight_changes.append((before - after).abs().mean().item())

   # Use MonitoredBioModule in your BioConverter
   # ... (similar to the CustomBioModule example above)

   # After training, you can analyze the weight changes
   import matplotlib.pyplot as plt

   plt.plot(bio_model.fc1.bio_mod.weight_changes)
   plt.title('Weight Changes Due to Rejuvenation')
   plt.xlabel('Rejuvenation Events')
   plt.ylabel('Average Absolute Weight Change')
   plt.show()

These advanced usage examples should help you customize and extend Bio Transformations to suit your specific needs. Remember to refer to the API documentation for detailed information on each class and method.