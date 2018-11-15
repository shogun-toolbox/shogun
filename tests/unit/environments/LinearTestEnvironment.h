/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: 2018 MikeLing, Viktor Gal, Sergey Lisitsyn, Heiko Strathmann, Gil Hoben
 */

#ifndef LINEARTESTENVIRONMENT_HPP
#define LINEARTESTENVIRONMENT_HPP

#include "GaussianCheckerboard.h"
#include "LinearRegressionDataGenerator.h"
#include <gtest/gtest.h>
#include <memory>

using namespace shogun;
using namespace std;
using ::testing::Environment;
class LinearTestEnvironment : public ::testing::Environment
{
public:
	virtual void SetUp()
	{
		sg_rand->set_seed(17);
		SGVector<float64_t> coefficients(1);
		float64_t bias = 2.0;
		coefficients[0] = 3.0;

		mBinaryLabelData = std::shared_ptr<GaussianCheckerboard>(
			new GaussianCheckerboard(100, 2, 2));
		// generate linear regression data y = 3x + 2
		one_dimensional_regression_data_with_bias = std::shared_ptr<LinearRegressionDataGenerator>(
				new LinearRegressionDataGenerator(100, coefficients, bias, 0.95));
		one_dimensional_regression_data = std::shared_ptr<LinearRegressionDataGenerator>(
				new LinearRegressionDataGenerator(100, coefficients, 0.0, 0.95));
	}

	std::shared_ptr<GaussianCheckerboard> getBinaryLabelData() const
	{
		return mBinaryLabelData;
	}

	std::shared_ptr<LinearRegressionDataGenerator> get_one_dimensional_regression_data(bool withBias) const
	{
		std::shared_ptr<LinearRegressionDataGenerator> data;
		if (withBias)
			data = one_dimensional_regression_data_with_bias;
		else
			data = one_dimensional_regression_data;
		return data;
	}

protected:
	std::shared_ptr<GaussianCheckerboard> mBinaryLabelData;
	std::shared_ptr<LinearRegressionDataGenerator> one_dimensional_regression_data;
	std::shared_ptr<LinearRegressionDataGenerator> one_dimensional_regression_data_with_bias;
};
#endif
