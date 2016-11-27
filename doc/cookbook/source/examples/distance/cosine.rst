==================
Cosine	 Distance
==================

The Cosine distance for real valued features is the similarity between two vectors by finding their angle.

.. math::

    d({\bf x},{\bf x'})= 1-\frac{{\bf x^\top x'}}{\sqrt{\sum_{i=1}^{d}{\bf {x_i}^2}\sum_{i=1}^{d}{\bf {x'_i}^2}}}

where :math:`\bf x` and :math:`\bf x'` are :math:`d` dimensional feature vectors.

-------
Example
-------

Imagine we have files with data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) as

.. sgexample:: cosine.sg:create_features

We create an instance of :sgclass:`CCosineDistance` by passing it :sgclass:`CDenseFeatures`.

.. sgexample:: cosine.sg:create_instance

The distance matrix can be extracted as follows:

.. sgexample:: cosine.sg:extract_distance

We can use the same instance with new :sgclass:`CDenseFeatures` to compute asymmetrical distance as follows:

.. sgexample:: cosine.sg:refresh_distance

----------
References
----------
:wiki:`Cosine_distance`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
