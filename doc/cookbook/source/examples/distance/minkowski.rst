====================
Minkowski Distance
====================

The Minkowski distance is a general class of distances for a :math:`R^n` feature space and is defined as

.. math::

    \big(\sum_{i=1}^{n} |{\bf x_i} - {\bf x'_i}|^k \big)^{\frac{1}{k}}

where :math:`\bf x` and :math:`\bf x'` are :math:`n` dimensional feature vectors.

:doc:`../distance/manhattan` and :doc:`../distance/euclidean`  are special cases for :math:`k=1` and :math:`k=2` respectively.


-------
Example
-------

Imagine we have files with data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) as

.. sgexample:: minkowski.sg:create_features

We create an instance of :sgclass:`CMinkowskiMetric` by passing it :sgclass:`CDenseFeatures`. We choose the order :math:`k` to be 3.	

.. sgexample:: minkowski.sg:create_instance

The distance matrix can be extracted as follows:

.. sgexample:: minkowski.sg:extract_distance

We can use the same instance with new :sgclass:`CDenseFeatures` to compute asymmetrical distance as follows:

.. sgexample:: minkowski.sg:refresh_distance

----------
References
----------
:wiki:`Minkowski_distance`


