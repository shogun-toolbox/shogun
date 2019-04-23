=============
Custom Kernel
=============

The Custom Kernel allows for custom user provided kernel matrices.

-------
Example
-------
In this example, we initialize a :sgclass:`GaussianKernel` and then use the kernel matrix to create a custom kernel.

Imagine we have files with data.
We create :sgclass:`DenseFeatures` (here 64 bit floats aka RealFeatures) as

.. sgexample:: custom_kernel_machine:create_features

We create an instance of :sgclass:`GaussianKernel`.
We initialize with :sgclass:`DenseFeatures` and then get the kernel matrix.

.. sgexample:: custom_kernel_machine:create_kernel

We use the provided kernel matrix to create custom kernels.

.. sgexample:: custom_kernel_machine:create_custom_kernel

We create an instance of :sgclass:`CLibSVM` with the custom kernel.

.. sgexample:: custom_kernel_machine:create_machine

Then we train the :sgclass:`CLibSVM` and we apply it to the test data, which gives the predicted :sgclass:`Labels`.

.. sgexample:: custom_kernel_machine:train_and_apply

Finally, we can evaluate the performance, e.g. using :sgclass:`CAccuracyMeasure`.

.. sgexample:: custom_kernel_machine:evaluate_accuracy

