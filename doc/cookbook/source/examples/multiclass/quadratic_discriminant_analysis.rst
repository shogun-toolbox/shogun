===============================
Quadratic Discriminant Analysis
===============================

Quadratic discriminant analysis (QDA) is used to separate measurements of two or more classes of objects by a quadric surface.
For QDA, the class label :math:`y` is assumed to be quadratic in the measurements of observations :math:`X`, i.e.:

.. math::

    \mathbf{x^{T}Ax} + \mathbf{b^{T}x} + c

QDA is a generalization of linear discriminant analysis (LDA). See Chapter 16 in :cite:`barber2012bayesian` for a detailed introduction.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CMulticlassLabels` as

.. sgexample:: quadratic_discriminant_analysis.sg:create_features


We create an instance of the :sgclass:`CQDA` classifier with feature matrix and label list.
:sgclass:`CQDA` also has two default arguments, to set tolerance used in training and mark whether to store the within class covariances

.. sgexample:: quadratic_discriminant_analysis.sg:create_instance

We run the train QDA algorithm and apply it to test data, which here gives :sgclass:`CMulticlassLabels`.

.. sgexample:: quadratic_discriminant_analysis.sg:train_and_apply

We can extract the mean vector of one class.
If we enabled storing covariance when creating instances, we can also extract the covariance matrix of the class:

.. sgexample:: quadratic_discriminant_analysis.sg:extract_mean_and_cov

We can evaluate test performance via e.g. :sgclass:`CMulticlassAccuracy`.

.. sgexample:: quadratic_discriminant_analysis.sg:evaluate_accuracy


----------
References
----------

:wiki:`Quadratic_classifier#Quadratic_discriminant_analysis`

:wiki:`Linear_discriminant_analysis`
