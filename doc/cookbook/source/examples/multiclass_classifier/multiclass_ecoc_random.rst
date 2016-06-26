=========================================
Multi-class Error-Correcting Output Codes
=========================================

ECOC (Error-Correcting Output Codes) is a multi-class learning strategy. ECOC trains :math:`L` binary classifers and transforms the results of the multiple classifications
into a matrix, which is called ECOC codebook. A decoder is applied to interpret the codebook, and to predict the labels of the samples.

The difference between ECOC and OvR/OvO strategies (`See multi-class linear machine cookbook <http://shogun.ml/cookbook/latest/examples/classifier/multiclass_linearmachine.html>`_)
is that in ECOC, :math:`L` is greater than class number :math:`K`, hence the training process is error-correcting.

There are multiple methods to encode or decode a codebook. In this example, we show how to apply random encoder and hamming distance decoder to multi-class dataset.

Codebooks can also be encoded as dense or sparse. For dense codebooks, only :math:`+1` and :math:`-1` are generated as labels for each sample in each binary classifier. In
sparse codebooks, :math:`+1`, :math:`-1` and :math:`0` are allowed, where :math:`0` labels the samples that are not classified.

In this examples, we use :sgclass:`CECOCRandomDenseEncoder` as encoder. :sgclass:`CECOCRandomSparseEncoder` can be applied for generating sparse codebooks.
More encoders and decoders are available in Shogun and can be passed to :sgclass:`CECOCStrategy` via the interface :sgclass:`CECOCEncoder` and :sgclass:`CECOCDecoder`.

See :cite:`escalera2010error` for a detailed introduction

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CMulticlassLabels` as

.. sgexample:: multiclass_ecoc_random.sg:create_features

We use :sgclass:`CLibLinear` as base classifier and create an instance of :sgclass:`CLibLinear`.
(`See the linear SVM cookbook <http://shogun.ml/cookbook/latest/examples/binary_classifier/linear_svm.html>`_)

.. sgexample:: multiclass_ecoc_random.sg:create_classifier

We initialize :sgclass:`CECOCStrategy` by specifying the random dense encoder and the decoder

.. sgexample:: multiclass_ecoc_random.sg:choose_strategy

We create an instance of the :sgclass:`CLinearMulticlassMachine` classifier by passing it the ECOC strategies, together with the dataset, binary classifier and the labels.

.. sgexample:: multiclass_ecoc_random.sg:create_instance

Then we train and apply it to test data, which here gives :sgclass:`CMulticlassLabels`.

.. sgexample:: multiclass_ecoc_random.sg:train_and_apply

We can evaluate test performance via e.g. :sgclass:`CMulticlassAccuracy`.

.. sgexample:: multiclass_ecoc_random.sg:evaluate_accuracy

----------
References
----------

.. bibliography:: ../../references.bib
    :filter: docname in docnames
