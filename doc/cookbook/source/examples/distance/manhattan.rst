==================
Manhattan Distance
==================

The Manhattan distance( :math:`L_1` distance ) for real valued features is the absolute difference between the components of two data points.

.. math::

    \sum_{i=0}^{d}|{\bf x_i}-{\bf x'_i}|

where :math:`\bf x` and :math:`\bf x'` are :math:`d` dimensional feature vectors.

-------
Example
-------

Imagine we have files with data. We create DenseFeatures (here 64 bit floats aka RealFeatures) as

.. sgexample:: manhattan.sg:create_features

We create an instance of :sgclass:`CManhattanDistance` by passing it :sgclass:`DenseFeatures`.

.. sgexample:: manhattan.sg:create_instance

The distance matrix can be extracted as follows:

.. sgexample:: manhattan.sg:extract_distance

We can use the same instance with new :sgclass:`DenseFeatures` to compute asymmetrical distance as follows:

.. sgexample:: manhattan.sg:refresh_distance

----------
References
----------
:wiki:`Manhattan_distance`

:wiki:`L1_distance`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
