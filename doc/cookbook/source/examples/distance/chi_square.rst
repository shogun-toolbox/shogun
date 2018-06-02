===================
Chi Square Distance
===================

The Chi Square Distance for real valued features x and x' extends the concept of :math:`\chi^{2}` metric to negative values.
This distance is calculated by the equation:

.. math::

    d(\bf{x},\bf{x'}) = \sum_{i=1}^{n}\frac{(x_{i}-x'_{i})^2}{|x_{i}|+|x'_{i}|} \quad \bf{x},\bf{x'} \in R^{n}

-------
Example
-------

We first create some sample data. So we instantiate CDenseFeatures containing the sample data.

.. sgexample:: chi_square.sg:create_features

We create an instance of :sgclass:`CChiSquareDistance` by passing it the sample data :sgclass:`CDenseFeatures`.

.. sgexample:: cosine.sg:create_instance

The distance matrix can be extracted as follows:

.. sgexample:: chi_square.sg:extract_distance

We can use the same instance with new :sgclass:`CDenseFeatures` to compute asymmetrical distance as follows:

.. sgexample:: chi_square.sg:refresh_distance

----------
References
----------
:wiki:`Chi-squared_test`
