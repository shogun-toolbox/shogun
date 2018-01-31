==================
Canberra Distance
==================

Canberra distance is a metric function used to calculate distance between two points in a vector space. It is a weighted Manhattan Distance. 
The Canberra distance :math:`d` between two :math:`n` dimensional feature vectors, :math:`\bf x` and :math:`\bf x'` is given by: 

.. math::

    d(\mathbf {x} ,\mathbf {x'} )=\sum _{i=1}^{n}{\frac {|x_{i}-x'_{i}|}{|x_{i}|+|x'_{i}|}}

where :math:`{\displaystyle \mathbf {x} =(x_{1},x_{2},\dots ,x_{n}){\text{ and }}\mathbf {x'} =(x'_{1},x'_{2},\dots ,x'_{n})}`.

-------
Example
-------

Imagine we have files with data. We create :sgclass:`CDenseFeatures` (here 64 bit floats aka RealFeatures) as

.. sgexample:: canberra.sg:create_features

We create an instance of :sgclass:`CCanberraMetric` by passing it two :sgclass:`CDenseFeatures` objects of whose features' pairwise Canberra Distance is to be calculated.

.. sgexample:: canberra.sg:create_instance

The distance matrix can be extracted as follows:

.. sgexample:: canberra.sg:extract_distance

We can use the same instance with new :sgclass:`CDenseFeatures` to compute asymmetrical distance as follows:

.. sgexample:: canberra.sg:refresh_distance

----------
References
----------
:wiki:`Canberra_distance`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
