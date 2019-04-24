=============
String Kernels
=============

A string kernel is a kernel function that operates on strings, i.e. finite sequences of symbols that do not need to be of the same length. It can be understood as a function that measures the similarity of a string pair. When string kernels are used in combination with kernelized learning algorithms such as support vector machines, they can become very useful tools for sequence classification (e.g. in text mining and gene analysis).

-------
Example
-------

Assume train_dna and test_dna each contains a list of dna sequences, we map them into string features.

.. sgexample:: string_kernels.sg:create_features

Now we will create different kernels that can take string features as input.

:sgclass:`CWeightedDegreeStringKernel`

.. sgexample:: string_kernels.sg:create_instance_1

:sgclass:`CWeightedDegreePositionStringKernel`

.. sgexample:: string_kernels.sg:create_instance_2

:sgclass:`CFixedDegreeStringKernel`

.. sgexample:: string_kernels.sg:create_instance_3

:sgclass:`CLocalityImprovedStringKernel`

.. sgexample:: string_kernels.sg:create_instance_4

:sgclass:`CLocalAlignmentStringKernel`

.. sgexample:: string_kernels.sg:create_instance_5

:sgclass:`CPolyMatchStringKernel`

.. sgexample:: string_kernels.sg:create_instance_6

We create an instance of the :sgclass:`CLibSVM` classifier by passing it regularization coefficient, kernel and labels.

.. sgexample:: string_kernels.sg:create_svm

Then we train and apply it to test data, which here gives :sgclass:`CBinaryLabels`.

.. sgexample:: string_kernels.sg:train_and_apply

We can evaluate test performance via :sgclass:`CROCEvaluation` and :sgclass:`CPRCEvaluation` using probabilities.

.. sgexample:: string_kernels.sg:evaluate

----------
References
----------
:wiki:`String_kernel`