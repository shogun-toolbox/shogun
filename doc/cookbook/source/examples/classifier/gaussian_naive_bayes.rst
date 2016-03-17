===================================
Gaussian Naive Bayes Classification
===================================

Gaussian Naive Bayes classifies data according to how well it aligns with the Gaussian distributions of several different classes.

The probability that some feature :math:`x_{i}` in the feature vector :math:`i` belongs to class :math:`c`, :math:`p(x_{i}|c)`, is given by

.. math::

  p(x_{i}|c)=\frac{1}{\sqrt{2\pi\sigma_{x,c}^{2}}}e^{-\frac{(x_{i}-\mu_{x,c})^{2}}{2\sigma_{x,c}^{2}}}
  
For each vector, the Gaussian Naive Bayes classifier chooses the class :math:`c` with maximal

.. math::

  p(c)\prod_{i}p(x_{i}|c)

See Chapter 10 in :cite:`barber2012bayesian` for a detailed introduction.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CMulticlassLabels` as

.. sgexample:: gaussiannaivebayes.sg:create_features

We create an instance of the :sgclass:`CGaussianNaiveBayes` classifier, passing it training data and the label.

.. sgexample:: gaussiannaivebayes.sg:create_instance

Then we run the train Gaussian Naive Bayes algorithm and apply it to the test data, which here gives `CMulticlassLabels`

.. sgexample:: gaussiannaivebayes.sg:train_and_apply

----------
References
----------
:wiki:`Naive_Bayes_classifier#Gaussian_naive_Bayes`
