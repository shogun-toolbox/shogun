============================
Passing features from matrix
============================

In this getting started guide, we will look at how to pass data random data to a matrix & pass it through a features object .

-------
Example
-------

We start off by creating a real matrix and inserting random values. 

.. sgexample:: passing_features_from_matrix.sg:create_matrix

Now we can pass the data from the vector to the features object.

.. sgexample:: passing_features_from_matrix.sg:create_features

----
NOTE
----

It is important to note that conventionally, data is stored in a row-major fashion, meaning two row vector of size 3.
However, in Shogun, this will be column vectors of dimension 2. This is beacuse each data sample is stored in a column-major fashion,
meaning each column here corresponds to an individual sample and each row in it to an atribute like BMI, Glucose concentration etc.
Also, it is essential to map the data types of the features object & vector i.e one cannot pass a vector of integers to a method that expects floats

----------
References
----------
:wiki:`Feature (machine learning)`

