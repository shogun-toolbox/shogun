===================
BrayCurtis Distance
===================

The BrayCurtis Distance or Sorensen Distance is similar to the Manhattan Distance with normalization.

.. math::

    \frac{\sum_{i=0}^{n}|{\bf x_i}-{\bf x'_i}|}{\sum_{i=0}^{n}|{\bf x_i}+{\bf x'_i}|}

where :math:`\bf x` and :math:`\bf x'` are :math:`n` dimensional feature vectors.

-------
Example
-------

Imagine we have files with data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) as

.. sgexample:: braycurtis.sg:create_features

We create an instance of :sgclass:`CBrayCurtisDistance` by passing it :sgclass:`CDenseFeatures`.

.. sgexample:: braycurtis.sg:create_instance

The distance matrix can be extracted as follows:

.. sgexample:: braycurtis.sg:extract_distance

We can use the same instance with new :sgclass:`CDenseFeatures` to compute asymmetrical distance as follows:

.. sgexample:: braycurtis.sg:refresh_distance

----------
References
----------
'BrayCurtis Distance <http://people.revoledu.com/kardi/tutorial/Similarity/BrayCurtisDistance.html>'

.. bibliography:: ../../references.bib
    :filter: docname in docnames