====================
K Nearest neighbours
====================

KNN classifies data according to the majority of labels in the nearest neighbourhood, according to some underlying distance function :math:`d(x,x')`.

For :math:`k=1`, the label for point :math:`x_i` is :math:`y_{k^*}`, where :math:`k^*=\argmin_n d(x_i, x_n)`.  

-------
Example
-------

Imagine you have some training and test data.

.. sgexample:: knn.sg:load_data

In order to run KNN, we need to choose a distance, for example CEuclideanDistance, or other sub-classes of CDistance. The distance needs the data we want to classify.

.. sgexample:: knn.sg:choose_distance

Once you have chosen a distance, create an instance of the CKNN classifier, passing it training data and labels

.. sgexample:: knn.sg:create_instance

Now we run the KNN algorithm and apply it to test data

.. sgexample:: knn.sg:train_and_apply
