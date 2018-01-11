============================================
Cross Validation on Multiple Kernel Learning
============================================

Cross Validation is a model validation technique whose purpose is to give an insight
on how the model we are testing will generalize to an independent dataset. Essentially,
it is based on training and test the model many times on different complementary partitions of the original
training dataset and then to combine the validation results (e.g. average) to estimate a
the performance of the final predictive model.

-------
Example
-------
We'll use as example a classification problem solvable by using :sgclass:`CMKLClassification`.
For the sake of brevity, we'll skip the initialization of features, kernels and so on
(see :doc:`../regression/multiple_kernel_learning` for a more complete example of MKL usage).

.. sgexample:: cross_validation_multiple_kernel_learning_weights_storage.sg:create_classifier

Firstly, we initialize a splitting strategy :sgclass:`CStratifiedCrossValidationSplitting`, which is needed
to divide the dataset into folds, and an evaluation criterium :sgclass:`CAccuracyMeasure`, to evaluate the
performance of the trained models. Secondly, we create the :sgclass:`CCrossValidation` instance.
We set also the number of cross validation's runs.

.. sgexample:: cross_validation_multiple_kernel_learning_weights_storage.sg:create_cross_validation

To observe also the partial folds' results, we create a cross validation's observer :sgclass:`CParameterObserverCV`
and then we register it into the :sgclass:`CCrossValidation` instance.

.. sgexample:: cross_validation_multiple_kernel_learning_weights_storage.sg:create_observer

Finally, we evaluate the model and get the results (aka a :sgclass:`CCrossValidationResult` instance).

.. sgexample:: cross_validation_multiple_kernel_learning_weights_storage.sg:evaluate_and_get_result

We get the :math:`mean` of all the evaluation results and its standard deviation :math:`stddev`.

.. sgexample:: cross_validation_multiple_kernel_learning_weights_storage.sg:get_results

We can get more information about the single cross validation's runs and folds by using the observer we set before, like the kernels' weights.
We get the :sgclass:`CMKLClassification` machine used during the first run and trained on the first fold.

.. sgexample:: cross_validation_multiple_kernel_learning_weights_storage.sg:get_fold_machine

Then, from the trained machine, we get the weights :math:`\mathbf{w}` of its kernels.

.. sgexample:: cross_validation_multiple_kernel_learning_weights_storage.sg:get_weights

----------
References
----------

:wiki:`Cross-validation_(statistics)`
