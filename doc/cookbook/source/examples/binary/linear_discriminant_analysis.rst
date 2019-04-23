=============================
Linear Discriminant Analysis
=============================

LDA learns a linear classifier via finding a projection matrix that maximally discriminates the provided classes. The learned linear classification rule is optimal under the assumption that both classes a gaussian distributed with equal co-variance. To find a linear separation :math:`{\bf w}` in training, the in-between class variance is maximized and the within class variance is minimized.
The projection matrix is computed by maximizing the following objective:

.. math::

	J({\bf w})=\frac{{\bf w^T} S_B {\bf w}}{{\bf w^T} S_W {\bf w}}

where :math:`{\bf S_B}` is between class scatter matrix and :math:`{\bf S_W}` is within class scatter matrix.
The above derivation of LDA requires the invertibility of the within class matrix. This condition however, is violated when there are fewer data-points than dimensions. In this case SVD is used to compute projection matrix using an orthonormal basis :math:`{\bf Q}`

.. math:: 

	{\bf W} := {\bf Q} {\bf{W^\prime}}

See Chapter 16 in :cite:`barber2012bayesian` for a detailed introduction.

-------
Example
-------

We create DenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`BinaryLabels` from files with training and test data.

.. sgexample:: linear_discriminant_analysis.sg:create_features

We create an instance of the :sgclass:`CLDA` classifier and set features and labels. By default, Shogun automatically chooses the decomposition method based on :math:`{N<=D}` or :math:`{N>D}`.

.. sgexample:: linear_discriminant_analysis.sg:create_instance

Then we train and apply it to test data, which here gives :sgclass:`BinaryLabels`.

.. sgexample:: linear_discriminant_analysis.sg:train_and_apply

We can extract weights :math:`{\bf w}`.

.. sgexample:: linear_discriminant_analysis.sg:extract_weights

We can evaluate test performance via e.g. :sgclass:`CAccuracyMeasure`.

.. sgexample:: linear_discriminant_analysis.sg:evaluate_accuracy

----------
References
----------
:wiki:`Linear_discriminant_analysis`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
