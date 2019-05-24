=================================
Observable Least Angle Regression
=================================

In this example, we show how to retrieve information from a Shogun model during training.
Here we train a Least Angle Regression model. If you need more details about the LARS,
have a look at its dedicated cookbook `example <../regression/least_angle_regression.html>`_.

-------
Example
-------

As usual, we first create the train and test features needed.

.. sgexample:: least_angle_regression.sg:create_features

Secondly, we do a round of preprocessing, by normalizing both the train and test features.

.. sgexample:: least_angle_regression:preprocess_features

We also create an instance of :sgclass:`CLeastAngleRegression` with some custom parameters
which will be our model.

.. sgexample:: least_angle_regression:create_instance

Then we create an observer object which is attached to the LARS object. Here
we are using the :sgclass:`CParameterObserverLogger` which basically prints
all information it gets from the machine during training on stdout.

.. sgexample:: least_angle_regression:create_observer

Finally, we train the model and we apply it to the test features.

.. sgexample:: least_angle_regression:train_and_apply

From the observer, we can also extract the observations :sgclass:`ObservedValue`
recorded during the training. Here we extract the last observation.

.. sgexample:: least_angle_regression:extract_last_observation

Given the observation, we can get its content (e.g its name, its description etc.)

.. sgexample:: least_angle_regression:read_observation_information

Moreover, we can also unsubscribe the observer from the LARS model when
it is no longer needed.

.. sgexample:: least_angle_regression:unsubscribe_observer
