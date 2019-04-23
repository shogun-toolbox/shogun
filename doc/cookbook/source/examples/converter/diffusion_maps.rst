================
Diffusion Maps
================

Diffusion Maps is a non-linear dimensionality reduction method that uses 
eigenfunctions of Markov matrices to diffusion maps for efficient 
representations of complex geometric structures.
The diffusion kernel :math: `k` must satisfy the following properties:

1. :math: `k` is symmetric :math: `{\bf k}(x, y) = {\bf k}(y, x)`
2. :math: `k` is positivity preserving :math: `{\bf k}(x, y) ≥ 0`


For more information see :cite:`Coifman-Lafon2006Diffusionmaps`.

-------
Example
-------

We create DenseFeatures (RealFeatures, here 64 bit float values).

.. sgexample:: diffusionmaps.sg:create_features

We create a :sgclass:`CDiffusionMaps` instance, and set its parameters. 

.. sgexample:: diffusionmaps.sg:set_parameters

Then we apply diffusion maps, which gives us distance embeddings.

.. sgexample:: diffusionmaps.sg:apply_convert

We can also extract the estimated feature_matrix.

.. sgexample:: diffusionmaps.sg:extract

----------
References
----------
:wiki:`Diffusion_map`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
