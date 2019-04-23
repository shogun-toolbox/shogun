==============================
Independent Component Analysis
==============================

Independent component analysis (ICA) separates a multivariate signal into additive subcomponents that are maximally independent.
It is typically used for separating superimposed signals.

The ICA algorithm presented here is fastICA, see :cite:`hyvarinen2000independent` for details.
There are many other ICA implementations, all based on :sgclass:`ICAConverter`

-------
Example
-------

Given a dataset which we assume consists of linearly mixed signals, we create DenseFeatures
(RealFeatures, here 64 bit float values).

.. sgexample:: independent_component_analysis_fast.sg:create_features

We create the :sgclass:`CFastICA` instance, and set its parameter for the iterative optimization.

.. sgexample:: independent_component_analysis_fast.sg:set_parameters

Then we apply ICA, which gives the unmixed signals.

.. sgexample:: independent_component_analysis_fast.sg:apply_convert

We can also extract the estimated mixing matrix.

.. sgexample:: independent_component_analysis_fast.sg:extract

:sgclass:`ICAConverter` supports inverse transformation. We can mix signals using the mixing matrix learnt before.

.. sgexample:: independent_component_analysis_fast.sg:inverse_transform

----------
References
----------
:wiki:`Independent_component_analysis`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
