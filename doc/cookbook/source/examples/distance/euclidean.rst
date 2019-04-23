==================
Euclidean Distance
==================

The Euclidean distance for real valued features is the square root of the sum of squared disparity between the corresponding feature dimensions of two data points.

.. math::

    d({\bf x},{\bf x'})= \sqrt{\sum_{i=0}^{d}|{\bf x_i}-{\bf x'_i}|^2}

where :math:`\bf x` and :math:`\bf x'` are :math:`d` dimensional feature vectors.

-------
Example
-------

Imagine we have files with data. We create DenseFeatures (here 64 bit floats aka RealFeatures) as

.. sgexample:: euclidean.sg:create_features

We create an instance of :sgclass:`EuclideanDistance` by passing it :sgclass:`DenseFeatures`.

.. sgexample:: euclidean.sg:create_instance

Distance matrix can be extracted as follows:

.. sgexample:: euclidean.sg:extract_distance

We can use the same instance with new :sgclass:`DenseFeatures` to compute distance.

.. sgexample:: euclidean.sg:refresh_distance

If desired, squared distance can be extracted like:

.. sgexample:: euclidean.sg:extract_sq_distance

----------
References
----------
:wiki:`Euclidean_distance`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
