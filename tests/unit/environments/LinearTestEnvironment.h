/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: 2016 MikeLing, Viktor Gal, Sergey Lisitsyn, Heiko Strathmann
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
		mOneDimensionalRegressionWithBiasData = std::shared_ptr<LinearRegressionDataGenerator>(
				new LinearRegressionDataGenerator(100, coefficients, bias, 0.95));
		mOneDimensionalRegressionData = std::shared_ptr<LinearRegressionDataGenerator>(
				new LinearRegressionDataGenerator(100, coefficients, 0.0, 0.95));
	}

	std::shared_ptr<GaussianCheckerboard> getBinaryLabelData() const
	{
		return mBinaryLabelData;
	}

	std::shared_ptr<LinearRegressionDataGenerator> getOneDimensionalRegressionData(bool withBias) const
	{
		std::shared_ptr<LinearRegressionDataGenerator> data;
		if (withBias)
			data = mOneDimensionalRegressionWithBiasData;
		else
			data = mOneDimensionalRegressionData;
		return data;
	}

protected:
	std::shared_ptr<GaussianCheckerboard> mBinaryLabelData;
	std::shared_ptr<LinearRegressionDataGenerator> mOneDimensionalRegressionData;
	std::shared_ptr<LinearRegressionDataGenerator> mOneDimensionalRegressionWithBiasData;
};
#endif
