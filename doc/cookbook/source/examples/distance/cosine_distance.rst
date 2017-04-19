===============
Cosine distance
===============

The Cosine distance is obtained by using the Cosine similarity (Orchini
similarity, angular similarity, normalized dot product), which
measures similarity between two vectors by finding their angle.

An extension to the Cosine similarity yields the Tanimoto coefficient.

:math:`d(\bf{x},\bf{x'}) = 1 - \frac{\sum_{i=1}^{n}\bf{x_{i}}\bf{x'_{i}}} {\sqrt{\sum_{i=1}^{n} x_{i}^2 \sum_{i=1}^{n} {x'}_{i}^2}} \quad x,x' \in R^{n}`
 
-------
Example
-------

We start by creating CDenseFeatures (here 64 bit floats aka RealFeatures) from files with training and test data.
.. sgexample:: cosine_distance.sg:create_features

Then, we create an instance of CosineDistance, passing it the training data.
.. sgexample:: cosine_distance.sg:create_distance

Subsequently, we retrieve the distance for the training data. After that we initialize both training and test data and retrieve the distance of the test data.
.. sgexample:: cosine_distance.sg:train_and_init

----------
References
----------
:wiki:`Cosine_similarity`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

