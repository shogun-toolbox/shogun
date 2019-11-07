==============================
Dense feature type dispatching
==============================

:sgclass:`DenseFeatures`, like those read from a :sgclass:`CCSVFIle`, can be read into memory in multiple primitive types, e.g. 32 bit or 64 bit floating point numbers.
All algorithms that inherit from :sgclass:`DenseRealDispatch` downstream can deal with multiples of such types, and they will carry out required computations in the corresponding primitive type.
Reducing from 64 bit float to 32 bit float can be beneficial if for example very large matrices have to be stored.

-------
Example
-------

Imagine we have files with training data. 
We create 64 bit and 32 bit float :sgclass:`DenseFeatures` of appropriate primitive type and also create :sgclass:`BinaryLabels` as

.. sgexample:: dense_dispatching.sg:create_features

We create an instance of a shogun algorithm (here :sgclass:`CLDA`) and provide labels.

.. sgexample:: dense_dispatching.sg:create_instance

We can train with any of the feature types as

.. sgexample:: dense_dispatching.sg:train_with_double

.. sgexample:: dense_dispatching.sg:train_with_float
