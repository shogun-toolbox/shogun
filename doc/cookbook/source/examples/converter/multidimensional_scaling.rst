========================
Multidimensional Scaling
========================

Multidimensional Scaling (MDS) is a set of data analysis methods, which allow one to infer the dimensions of the perceptual space of subjects.
The entering data are typically a measure of the global similarity or dissimilarity of objects, called proximities.
The outcome is a spatial configuration in which the objects are represented as points arranged in such a way, that their distances correspond to their proximities.

This example features classical MDS, a subset of metric MDS, where the distances in the transformed space preserve the intervals and ratios between the proximities as good as possible. In contrast, nonmetric MDS only reflects the order of the proximities.

See :cite:`JEDM:JEDM277` for a detailed introduction.

In order to find the coordinates :math:`(x_n, y_n)` of :math:`N` items given the proximity matrix :math:`D` with entries the distances :math:`d_{ij}=\sqrt{(x_i -x_j)^2 + (y_i - y_j)^2}` between pairs of items, classical MDS minimizes the following loss function:

.. math::
    \frac{\sum_{ij}^{N} (b_{ij} -d_{ij})^2}{\sum_{ij}^{N} b_{ij}^2}

where :math:`b_{ij}` are the terms of matrix :math:`B`, derived from matrix :math:`D` by applying double centering. This problem is tractable and solved using eigenvalue decomposition from :math:`B=X X^\top`

:cite:`Silva2004SparseMS` introduced a two-step, computationally efficient approximation of classical MDS for use when the number of data points is too large. First, it uses a subset of the initial points as landmarks to apply classical MDS and then, it uses a distance-based triangulation procedure to determine the positions of the remaining points.

-------
Example
-------

Imagine we have a file with data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) as

.. sgexample:: multidimensional_scaling:create_features

We create an instance of :sgclass:`CEuclideanDistance` by passing it :sgclass:`CDenseFeatures`. We calculate the distance matrix.

.. sgexample:: multidimensional_scaling:calculate_distances_before

We create an instance of :sgclass:`MultidimensionalScaling` . The MDS space is set to be two-dimensional and we choose to disable the Sparse MDS using landmark points technique. We apply MDS to our features. :sgclass:`MultidimensionalScaling` uses Euclidean distance as proximity by default.

.. sgexample:: multidimensional_scaling:apply_mds_no_landmark

We apply MDS leaving the Sparse MDS using landmark points technique activated.
 
.. sgexample:: multidimensional_scaling:apply_mds_landmark

We calculate the Euclidean distance matrices between our features in the MDS space for both cases.

.. sgexample:: multidimensional_scaling:calculate_distances_after

----------
References
----------
:wiki:`Multidimensional_scaling`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

