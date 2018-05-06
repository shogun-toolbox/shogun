// tests/unit/stanmath/StanPerceptronTest_unittest.cc
#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <stan/math.hpp>
using namespace std;
using namespace Eigen;
using namespace stan::math;

/*
This test will implement a perceptron using stan math
The perceptron will have one input, the column vector [1 1 1]^T
Then, it will have 2x3 Matrix of weights that it will learn
Then, the output is a 2x1 column vector
In this example, we want to learn the weights W such that the square
Error loss from the output of the perceptron to [1 1]^T is minimized.
Since we can find weights from [1 1 1]^T to [1 1]^T in a perceptron,
this error should be very close to zero after 100 epochs.
*/
TEST(StanPerceptronTest, sample_perceptron)
{
	// Initialize the Input Vector
	Matrix<var, 3, 1> inp;
	inp(0, 0) = 1;
	inp(1, 0) = 1;
	inp(2, 0) = 1;

	// Randomly Initialize the weights on the perceptron
	std::random_device rd{};
	std::mt19937 gen{rd()};
	normal_distribution<> d{0, 1};
	Matrix<var, 2, 3> W1;
	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			W1(i, j) = 0.01 * d(gen);
		}
	}

	// Define the outputs of the neural network
	Matrix<var, 2, 1> outputs;

	double learning_rate = 0.1;
	double last_error = 0;
	for (int epoch = 0; epoch < 100; ++epoch)
	{
		var error = 0;
		outputs = W1 * inp;
		for (int i = 0; i < 2; ++i)
		{
			error += (outputs(i, 0) - 1) * (outputs(i, 0) - 1);
		}
		error.grad();

		// Now use gradient descent to change the weights
		for (int i = 0; i < 2; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				W1(i, j) = W1(i, j) - learning_rate * W1(i, j).adj();
			}
		}

		// Store the value of current error in last_error
		last_error = value_of(error);
	}

	// Error should be very close to 0.0
	EXPECT_NEAR(last_error, 0.0, 1e-6);
}
