============================================
Cross Validation on Multiple Kernel Learning
============================================
In this example, we illustrate how to use cross-validation with multiple kernel learning (MKL)

For more information on cross-validation, check :doc:`./cross_validation`.

-------
Example
-------
We'll use as example a classification problem solvable by using :sgclass:`MKLClassification`.
For the sake of brevity, we'll skip the initialization of features, kernels and so on
(see :doc:`../regression/multiple_kernel_learning` for a more complete example of MKL usage).

.. sgexample:: cross_validation_multiple_kernel_learning_weights_storage.sg:create_classifier

Firstly, we initialize a splitting strategy :sgclass:`StratifiedCrossValidationSplitting`, which is needed
to divide the dataset into folds, and an evaluation criterium :sgclass:`CAccuracyMeasure`, to evaluate the
performance of the trained models. Secondly, we create the :sgclass:`CrossValidation` instance.
We set also the number of cross validation's runs.

.. sgexample:: cross_validation_multiple_kernel_learning_weights_storage.sg:create_cross_validation

To observe also the partial folds' results, we create a cross validation's observer :sgclass:`ParameterObserverCV`
and then we register it into the :sgclass:`CrossValidation` instance.

.. sgexample:: cross_validation_multiple_kernel_learning_weights_storage.sg:create_observer

Finally, we evaluate the model and get the results (aka a :sgclass:`CrossValidationResult` instance).

.. sgexample:: cross_validation_multiple_kernel_learning_weights_storage.sg:evaluate_and_get_result

We get the :math:`mean` of all the evaluation results and its standard deviation :math:`stddev`.

.. sgexample:: cross_validation_multiple_kernel_learning_weights_storage.sg:get_results

We can get more information about the single cross validation's runs and folds by using the observer we set before, like the kernels' weights.
We get the :sgclass:`MKLClassification` machine used during the first run and trained on the first fold.

.. sgexample:: cross_validation_multiple_kernel_learning_weights_storage.sg:get_fold_machine

Then, from the trained machine, we get the weights :math:`\mathbf{w}` of its kernels.

.. sgexample:: cross_validation_multiple_kernel_learning_weights_storage.sg:get_weights

----------
References
----------

:wiki:`Cross-validation_(statistics)`
