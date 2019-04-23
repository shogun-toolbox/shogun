========
Pipeline
========

:sgclass:`CPipeline` is a machine that chains multiple transformers and machines.
It consists of a sequence of transformers as intermediate stages of training or testing and a machine as the final stage.
Features are transformed by transformers and fed into the next stage sequentially.

-------
Example
-------
Imagine we have files with training data. We create :sgclass:`DenseFeatures` (here 64 bit floats aka RealFeatures) as

.. sgexample:: pipeline:create_features

To project the feature to a lower dimension, we create an instance of transformers :sgclass:`CPruneVarSubMean` and :sgclass:`CPCA`.

.. sgexample:: pipeline:create_transformers

We then perform clustering using :sgclass:`KMeans`.

.. sgexample:: pipeline:create_machine

To chain these algorithms, we create an instance of :sgclass:`CPipeline`.

.. sgexample:: pipeline:create_pipeline

Then we train and apply the pipeline.

.. sgexample:: pipeline:train_and_apply

