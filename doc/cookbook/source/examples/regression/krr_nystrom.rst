===============================
Nyström Kernel Ridge Regression
===============================

The Nyström method is a technique for reducing  the computational load of
kernel methods by replacing the kernel matrix with a low rank approximation.

The approximation is achieved by projecting the data matrix on a subset of data
points, resulting in a linear system that is cheaper to solve. When applied to
ridge regression, the following approximate system is solved

.. math::
    {\bf \alpha} = (\tau K_{m,m} + K_{m,n}K_{n,m})^+K_{m,n} {\bf y}

where :math:`K_{n,m}` is a submatrix of the kernel matrix :math:`K` containing
all :math:`n` rows and the :math:`m` columns corresponding to the :math:`m`
chosen training examples, :math:`K_{m,n}` is its transpose and :math:`K_{m,m}`
is the submatrix with the m rows and columns corresponding to the training
examples chosen. :math:`+` indicates the Moore-Penrose pseudoinverse. The
computational complexity is :math:`O(m^3 + m^2n)` versus :math:`O(n^3)` for
ordinary kernel ridge regression. The memory requirements are :math:`O(m^2 + nm)`
versus :math:`O(n^2)`.

Several ways to subsample columns of the data matrix have been proposed. The
default method is to subsample columns uniformly.

The Nyström method has been shown empirically to give good results while
substantially increasing performance.

See :cite:`williams2001using` or :cite:`rudi2015less` for more details.

-------
Example
-------

The API is very similar to :sgclass:`CKernelRidgeRegression`. We create
:sgclass:`CDenseFeatures` (here 64 bit floats aka RealFeatures) and
:sgclass:`CRegressionLabels` as

.. sgexample:: krr_nystrom.sg:create_features

Choose an appropriate :sgclass:`CKernel` and instantiate it. Here we use a
:sgclass:`CGaussianKernel`.

.. sgexample:: krr_nystrom.sg:create_appropriate_kernel

We create an instance of :sgclass:`CKRRNystrom` by passing it the ridge penalty
:math:`\tau`, the number of data columns to subsample :math:`m`, the kernel and
the labels.

.. sgexample:: krr_nystrom.sg:create_instance

Then we train the regression model and apply it to test data to get the
predicted :sgclass:`CRegressionLabels`.

.. sgexample:: krr_nystrom.sg:train_and_apply

After training, we can extract :math:`\alpha`.

.. sgexample:: krr_nystrom.sg:extract_alpha

Finally, we can evaluate the :sgclass:`CMeanSquaredError`.

.. sgexample:: krr_nystrom.sg:evaluate_error

----------
References
----------
.. bibliography:: ../../references.bib
    :filter: docname in docnames
