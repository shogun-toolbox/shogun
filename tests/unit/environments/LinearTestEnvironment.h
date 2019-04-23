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

#include <random>

using namespace shogun;
using namespace std;
using ::testing::Environment;
class LinearTestEnvironment : public ::testing::Environment
{
public:
	virtual void SetUp()
	{
		std::mt19937_64 prng(125);
		SGVector<float64_t> coefficients(1);
		float64_t bias = 2.0;
		coefficients[0] = 3.0;

		mBinaryLabelData = std::make_shared<GaussianCheckerboard>(100, 2, 2, prng);
		// generate linear regression data y = 3x + 2
		one_dimensional_regression_data_with_bias = std::make_shared<LinearRegressionDataGenerator>(
			100, coefficients, bias, 0.95);
		one_dimensional_regression_data = std::make_shared<LinearRegressionDataGenerator>(
				100, coefficients, 0.0, 0.95);
	}

	auto getBinaryLabelData() const
	{
		return mBinaryLabelData;
	}

	auto get_one_dimensional_regression_data(bool withBias) const
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
