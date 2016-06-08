=========================
Bray–Curtis dissimilarity
=========================

The Bray–Curtis dissimilarity (Sorensen distance) is similar to the
Manhattan distance with normalization.

:math:`d(\bf{x},\bf{x}') = \frac{\sum_{i=1}^{n}|x_{i}-x'_{i}|} {\sum_{i=1}^{n}|x_{i}+x'_{i}|} \ 
\quad x,x' \in R^{n}`
 
-------
Example
-------

We start by creating CDenseFeatures (here 64 bit floats aka RealFeatures) from files with training and test data.
.. sgexample:: braycurtis_dissimilarity.sg:create_features

Then, we create an instance of BrayCurtisDistance, passing it the training data.
.. sgexample:: braycurtis_dissimilarity.sg:create_distance

Subsequently, we retrieve the distance for the training data. After that we initialize both training and test data and retrieve the distance of the test data.
.. sgexample:: braycurtis_dissimilarity.sg:train_and_init

----------
References
----------
:wiki:`Bray–Curtis_dissimilarity`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

