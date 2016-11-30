==================
Cosine	 Distance
==================

The Cosine distance for real valued features x and x' is the similarity as measured by their angle.

.. math::

    1-\frac{{\bf x^\top x'}}{\Vert \bf{x}\Vert_2 \Vert \bf{x'}\Vert_2 }

where where :math:`\Vert \cdot\Vert_2` is the Euclidean norm.

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
