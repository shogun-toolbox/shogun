==================================================
Dense Dispatching for Linear Discriminant Analysis
==================================================

Some algorithms like Linear Discriminant Analysis, Least Angle Regression, etc. in shogun support training with features of multiple primitive type.

-------
Example
-------

Imagine we have files with training data. We create 64 bit and 32 bit float CDenseFeatures of appropriate primitive type and also create :sgclass:`CBinaryLabels` as

.. sgexample:: dense_dispatching.sg:create_features

We create an instance of :sgclass:`CLDA` and provide labels.

.. sgexample:: dense_dispatching.sg:create_instance

We train with 64 bit RealFeatures as

.. sgexample:: dense_dispatching.sg:train_with_double

We can train the same instance with 32 bit RealFeatures as

.. sgexample:: dense_dispatching.sg:train_with_float
