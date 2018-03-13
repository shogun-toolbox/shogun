============================================
Cross Validation on a Linear SVM Model
============================================

Cross Validation is a type of model validation technique for figuring out how the
results from a statistical analysis will generalize to an independent data set. It
essentially involves training the model on a subset of the data, and testing on the
remainder, then repeating this process for several complementary partitions and then
combining all the validation results generated (For instance, an average) to estimate
a performance for the final predictive model.

One type of cross validation is known as :math:`k-` cross validation.
For instance, Suppose you have the training data :math:`\mathbf{x}_1,\mathbf{x}_2, ..., \mathbf{x}_{10n}`
for a positive integer :math:`n` , then :math:`10-` cross validation might for instance partition the data into
:math:`P_1, P_2, ..., P_n` each with :math:`10` data points. Then, the model would be trained on
:math:`P_1, P_2, ...,P_{n-1}` and test it on :math:`P_n` to get accuracy :math:`a_1` . Then the model
would train again from scratch on :math:`P_1, P_2, ...,P_{n-2}, P_{n}` and is tested on :math:`P_{n-1}` to get
:math:`a_2`.  This process is repeated until we have :math:`a_1, a_2, ..., a_n`, at which we take their average,
and assign that as the accuracy of this mode.


In this tutorial, we will be doing :math:`K-` Cross Validation on a linear support vector machine (SVM) model.

-------
Example
-------
We'll use as example a classification problem solvable by using Linear SVM or :sgclass:`CLibLinear` .
(see :doc:`../binary/linear_support_vector_machine` for a more complete example of Linear SVM usage).


Firstly, we import the data and the labels where the lablels are binary.

.. sgexample:: cross_validation_linear_svm.sg:create_features

Next, we set the required parameters needed by the Linear SVM Mode.

.. sgexample:: cross_validation_linear_svm.sg:set_parameters

Now,we initialize a splitting strategy :sgclass:`CStratifiedCrossValidationSplitting`, which is needed
to divide the dataset into folds for the :math:`k-` cross validation (In this case, we make :math:`k=2`),
and an evaluation criterion :sgclass:`CAccuracyMeasure`, to evaluate the performance of the trained models. then, we create the
:sgclass:`CCrossValidation` instance. We also set the number of cross validation's runs.

.. sgexample:: cross_validation_linear_svm.sg:create_cross_validation

Now, we initialize the instance of our linear model and feed it out parameters.

.. sgexample:: cross_validation_linear_svm.sg:create_instance

Finally, we evaluate the model and get the results (aka a :sgclass:`CCrossValidationResult` instance).

.. sgexample:: cross_validation_linear_svm.sg:evaluate_and_get_result

We get the :math:`mean` of all the evaluation results and its standard deviation :math:`stddev`.

.. sgexample:: cross_validation_linear_svm.sg:get_results

----------
References
----------

:wiki:`Cross-validation_(statistics)`
