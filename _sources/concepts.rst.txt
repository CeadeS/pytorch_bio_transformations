Biological Concepts
===================

Bio Transformations is built on several key concepts from neuroscience. This page provides an overview of these concepts and how they relate to our implementation in artificial neural networks.

Synaptic Diversity
------------------

In biological neural networks, synapses (the connections between neurons) exhibit a wide range of properties and behaviors. This diversity is crucial for the complex information processing capabilities of the brain.

1. Diversity in Synaptic Plasticity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Biological Concept**: In the brain, not all synapses change their strength at the same rate or in the same way. Some synapses are more plastic (changeable) than others, and this plasticity can vary over time.

**Implementation**: The `fuzzy_learning_rates()` method in our `BioModule` applies different learning rates to different "synapses" (weights) in the artificial neural network. This mimics the diverse plasticity observed in biological synapses.

2. Spontaneous Spine Remodeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Biological Concept**: Dendritic spines, the primary sites of excitatory synapses in the brain, are dynamic structures. They can form, change shape, and disappear over time, even in the absence of explicit learning signals.

**Implementation**: The `rejuvenate_weights()` method simulates this by randomly reinitializing certain weights in the network. This allows for the "formation" of new connections and the "pruning" of others, similar to spine remodeling in the brain.

3. Multi-synaptic Connectivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Biological Concept**: In biological neural networks, multiple synaptic connections often exist between the same pair of neurons. This redundancy can enhance the reliability and flexibility of neural circuits.

**Implementation**: The `add_weight_splitting_step()` method implements this concept by allowing multiple "synapses" (sub-weights) to exist for each connection in the artificial neural network.

Homeostatic Plasticity
----------------------

**Biological Concept**: Neurons have mechanisms to maintain their activity levels within a functional range, preventing over-excitation or complete silencing. This is crucial for the stability of neural circuits.

**Implementation**: The `scale_grad()` method implements a form of homeostatic plasticity by scaling the gradients of the weights. This helps maintain overall network stability while still allowing for learning.

Volume-Dependent Plasticity
---------------------------

**Biological Concept**: In the brain, the size (volume) of a synapse is often correlated with its strength and stability. Larger synapses tend to be more stable but less plastic, while smaller synapses are more dynamic.

**Implementation**: The `volume_dependent_lr()` method implements this concept by adjusting learning rates based on the magnitude (analogous to volume) of the weights. Larger weights have smaller, less variable learning rates, while smaller weights have larger, more variable learning rates.

Dale's Principle
----------------

**Biological Concept**: Dale's principle states that a neuron releases the same neurotransmitter(s) at all of its synapses. This introduces certain constraints on how neurons can influence each other.

**Implementation**: The `enforce_dales_principle()` method ensures that all outgoing weights from a given artificial "neuron" have the same sign, mimicking the constraints imposed by Dale's principle in biological neural networks.

By incorporating these biological concepts, Bio Transformations aims to create artificial neural networks that capture some of the complex dynamics observed in biological brains. This bio-inspired approach has the potential to enhance the learning, adaptability, and robustness of artificial neural networks.