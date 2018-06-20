================
Diffusion Maps
================

Diffusion Maps is a non-linear dimensionality reduction method that uses 
eigenfunctions of Markov matrices to diffusion maps for efficient 
representations of complex geometric structures.

For more information see :cite:`Coifman-Lafon2006Diffusionmaps`.

-------
Example
-------

we create CDenseFeatures (RealFeatures, here 64 bit float values).

.. sgexample:: diffusionmaps.sg:create_features

We create the :sgclass:`CDiffusionMaps` instance, and set its parameters. The diffusion kernel must satisfy
the following properties:

1. :math: `k` is symmetric :math: `{\bf k}(x, y) = {\bf k}(y, x)`
2. :math: `k` is positivity preserving :math: `{\bf k}(x, y) ≥ 0`

.. sgexample:: diffusionmaps.sg:set_parameters

Then we apply DiffusionMaps, which gives us the distance embeddings.

.. sgexample:: diffusionmaps.sg:apply_convert

We can also extract the estimated feature_matrix.

.. sgexample:: diffusionmaps.sg:extract

----------
References
----------
:wiki:`Diffusion_map`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
