bio\_transformations package
============================

Overview
--------

The `bio_transformations` package implements biologically inspired modifications to artificial neural networks. It enhances the learning capabilities of neural networks by mimicking the plasticity and stability characteristics observed in biological synapses.

Key Components
--------------

1. :ref:`BioConverter <bio_converter>`

   The main interface for converting standard PyTorch modules to bio-inspired versions.

2. :ref:`BioModule <bio_module>`

   The core class implementing biologically inspired modifications.

Submodules
----------

.. _bio_converter:

bio\_transformations.bio\_converter module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: bio_transformations.bio_converter
   :members:
   :undoc-members:
   :show-inheritance:

The `BioConverter` class is responsible for converting standard PyTorch modules to BioNet modules with bio-inspired modifications. It provides methods for:

- Converting entire model architectures
- Configuring bio-inspired parameters
- Applying weight splitting techniques

.. _bio_module:

bio\_transformations.bio\_module module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: bio_transformations.bio_module
   :members:
   :undoc-members:
   :show-inheritance:

The `BioModule` class implements the core biologically inspired modifications. It includes the following key functions:

Synaptic Plasticity Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :func:`~bio_transformations.bio_module.BioModule.volume_dependent_lr`: Implements learning rates based on weight magnitude.
- :func:`~bio_transformations.bio_module.BioModule.fuzzy_learning_rates`: Applies diverse learning rates to individual synapses.
- :func:`~bio_transformations.bio_module.BioModule.crystallize`: Simulates synaptic stabilization over time.

Structural Plasticity Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :func:`~bio_transformations.bio_module.BioModule.rejuvenate_weights`: Mimics spine turnover in biological networks.

Homeostatic Plasticity Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :func:`~bio_transformations.bio_module.BioModule.scale_grad`: Implements synaptic scaling for overall network stability.

Additional Functions
~~~~~~~~~~~~~~~~~~~~

- :func:`~bio_transformations.bio_module.BioModule.l1_reg`: Computes L1 regularization of module parameters.
- :func:`~bio_transformations.bio_module.BioModule.enforce_dales_principle`: Enforces Dale's principle on module weights.
- :func:`~bio_transformations.bio_module.BioModule.dalian_network_initialization`: Initializes network weights according to Dale's principle.

For detailed information on each function, please refer to the individual function documentation.

Extending BioModule
~~~~~~~~~~~~~~~~~~~

To add a new biologically motivated function:

1. Implement the new function in the `BioModule` class.
2. Add the function name to the `exposed_functions` list.
3. Update `__init__` methods if new parameters are required.
4. Create a test case in `test_biomodule.py`.
5. Update this documentation to include details about the new function.

Module contents
---------------

.. automodule:: bio_transformations
   :members:
   :undoc-members:
   :show-inheritance:

For a practical guide on using these components, please refer to the :doc:`tutorials` section.