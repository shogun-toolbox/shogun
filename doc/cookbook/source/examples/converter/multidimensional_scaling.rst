========================
Multidimensional Scaling
========================

Multidimensional Scaling (MDS) is a set of data analysis methods, which allow one to infer the dimensions of the perceptual space of subjects. The entering data are typically a measure of the global similarity or dissimilarity of objects, called proximities. The outcome is a spatial configuration in which the objects are represented as points arranged in such a way, that their distances correspond to their proximities.

This example features classical MDS, a subset of metric MDS, where the distances in the transformed space preserve the intervals and ratios between the proximities as good as possible. In contrast, nonmetric MDS only reflects the order of the proximities.

See :cite:`Wickelmaier03anintroduction` for a detailed introduction.

In order to find the coordinates :math:`(x_n, y_n)` of :math:`N` items given the proximity matrix :math:`D` with entries the distances :math:`d_{ij}=\sqrt{(x_i -x_j)^2 + (y_i - y_j)^2}` between pairs of items, classical MDS minimizes the following loss function:

.. math::
    Strain_D(x_1, x_2, \cdots, x_N ) = \frac{\sum_{ij} (b_{ij} -d_{ij})^2}{\sum_{ij} b_{ij}^2}

where :math:`b_{ij}` are the terms of matrix :math:`B`, derived from matrix :math:`D` by applying double centering. This problem is tractable and solved using eigenvalue decomposition from :math:`B=X X'`

:cite:`Silva2004SparseMS` introduced a two-step, computationally efficient approximation of classical MDS for use when the number of data points is too large. First, it uses a subset of the initial points as landmarks to apply classical MDS and then, it uses a distance-based triangulation procedure to determine the positions of the remaining points.

-------
Example
-------

Imagine we have a file with data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) as

.. sgexample:: multidimensional_scaling:create_features

We calculate the Euclidean distance between our features.

.. sgexample:: multidimensional_scaling:calculate_distances_before

We create an instance of :sgclass:`MultidimensionalScaling` . The MDS space is set to be two-dimensional and we choose to disable the Sparse MDS using landmark points technique. We apply MDS to our features. :sgclass:`MultidimensionalScaling` uses Euclidean distance as proximity by default.

.. sgexample:: multidimensional_scaling:apply_mds

We calculate the Euclidean distance between our features in the MDS space.

.. sgexample:: multidimensional_scaling:calculate_distances_after

We calculate distance matrices before and after application of MDS.

.. sgexample:: multidimensional_scaling:calculate_distance_matrices




----------
References
----------
:wiki:`Multidimensional_scaling`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

