=======================
Passing data to Shogun
=======================

In this getting started guide, we will look at how to pass data random data to a matrix & pass it through a features object . We will create a matrix as shown :

+-------------+-------------+-----------+-----------+
|  	      |  Sample 1   |  Sample 2 |  Sample 3 |
|   	      |             |           |           |
+=============+=============+===========+===========+
| Feature 1   | 1	    | 2	        | 3	    |
+-------------+-------------+-----------+-----------+
| Feature 2   | 4           | 5         | 6         |
+-------------+-------------+-----------+-----------+

Then, we extract features for example 1 (values 1 and 4)


-------
Example
-------

We start off by creating a real matrix and inserting values 1 through 6. 

.. sgexample:: features.sg:create_matrix

Now we can pass the data from the matrix to the features object.

.. sgexample:: features.sg:create_features

We've got a features object containg our data , so we can now extract the features
for Sample 1.

.. sgexample:: features.sg:get_features

This vector now contains values 1 and 4.

----
NOTE
----

It is important to note that conventionally, data is stored in a row-major fashion,
meaning two row vector of size 3.
However, in Shogun, this will be column vectors of dimension 2. 

Note that data is stored in column major format, i.e. each column of the matrix corresponds to
an observation / feature vector, where each vector consists of a number of variables that is equal
to the number of rows of the matrix.

Also, it is essential to map the data types of the features object & vector i.e one cannot pass
a vector of integers to a method that expects floats. 
Note that features are type-safe, e.g. you can only pass 64 bit floating point numbers to :sgclass:`RealFeatures` .

----------
References
----------
:wiki:`Feature (machine learning)`

:wiki:`Row- and column-major order`

