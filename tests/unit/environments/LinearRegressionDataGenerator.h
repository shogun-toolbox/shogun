/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

// Combines GaussianCheckerboard.h and RegressionTestEnvironment.h
// GaussianCheckerboard only generates binary labels for classification
// RegressionTestEnvironment.h is inflexible with the dataset size
// and setting slopes and biases
// LinearRegressionDataGenerator combines the ideas of both classes
// and makes them more flexible

#ifndef LINEARREGRESSIONDATAGENERATOR_HPP
#define LINEARREGRESSIONDATAGENERATOR_HPP

#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/labels/RegressionLabels.h>


using namespace shogun;

/** @brief This class generates linear regression data
 * to be used in a testing environment:
 *
 *  {y} =X{\boldsymbol {\beta }}
 *
 * A bias value can be optionally set.
 * */

class LinearRegressionDataGenerator
{
public:
	/** constructor
	 * @param num_sample: nmber of random samples to be
	 *  generated
	 *  @param coefficient_values: SGVector with m coefficients
	 *  @param bias_value: bias value, should be set to 0
	 *  if it is not required
	 *  @param train_split: a float from 0. to 1. to set
	 *  the train size relative to the size of the generated
	 *  dataset
	 * */
	LinearRegressionDataGenerator(
			const int32_t num_samples,
			const SGVector<float64_t>& coefficient_values,
			const float64_t bias_value,
			const float64_t train_split)
	{
		std::mt19937_64 prng(77);

		n_dim = coefficient_values.size();
		bias = bias_value;

		train_size = static_cast<int32_t>(num_samples * train_split);
		test_size = num_samples - train_size;

		coefficients = SGVector<float64_t>(coefficient_values);
		SGMatrix<float64_t> feat_train_data = DataGenerator::generate_gaussians(
				train_size, 1, n_dim, prng);
		SGMatrix<float64_t> feat_test_data = DataGenerator::generate_gaussians(
				test_size, 1, n_dim, prng);

		SGVector<float64_t> label_train_data =
				linalg::matrix_prod(feat_train_data, coefficients, true);

		SGVector<float64_t> label_test_data =
				linalg::matrix_prod(feat_test_data, coefficients, true);

		linalg::add_scalar(label_train_data, bias);
		linalg::add_scalar(label_test_data, bias);

		features_train = std::make_shared<DenseFeatures<float64_t>>(feat_train_data);
		features_test = std::make_shared<DenseFeatures<float64_t>>(feat_test_data);

		labels_train = std::make_shared<RegressionLabels>(label_train_data);
		labels_test = std::make_shared<RegressionLabels>(label_test_data);

	}

	/** destructor */
	~LinearRegressionDataGenerator()
	{
	}

	/** gets the traning features */
	auto get_features_train() const
	{
		return features_train;
	}

	/** gets the test features */
	auto get_features_test() const
	{
		return features_test;
	}

	/** get the test labels */
	auto get_labels_train() const
	{
		return labels_train;
	}

	/** gets the traning labels */
	auto get_labels_test() const
	{
		return labels_test;
	}

	/** returns the size of train set */
	int32_t get_train_size() const
	{
		return train_size;
	}

	/** returns the size of the test set */
	int32_t get_test_size() const
	{
		return test_size;
	}

	/** returns the bias value */
	float64_t get_bias() const
	{
		return bias;
	}

	/** returns all coefficients */
	SGVector<float64_t> get_coefficients() const
	{
		return coefficients;
	}

	/** returns the coefficient at index i
	 * @param i: coefficient index
	 * */
	float64_t get_coefficient(index_t i) const
	{
		return coefficients[i];
	}

protected:
	// data for training
	std::shared_ptr<DenseFeatures<float64_t>> features_train;

	// data for testing
	std::shared_ptr<DenseFeatures<float64_t>> features_test;

	// training label
	std::shared_ptr<RegressionLabels> labels_train;

	// testing label
	std::shared_ptr<RegressionLabels> labels_test;

	// the size of generated of the training set
	int32_t train_size;

	// the size of generated of the test set
	int32_t test_size;

	// number of dimensions of data (excluding bias)
	int32_t n_dim;

	// bias value
	float64_t bias;

	// coefficients {x[0], x[1], ..., x[m]}
	SGVector<float64_t> coefficients;
};
#endif // LINEARREGRESSIONDATAGENERATOR_HPP
