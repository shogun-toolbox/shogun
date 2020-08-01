/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <gtest/gtest.h>
#include <shogun/base/ShogunEnv.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/loss/SquaredLoss.h>
#include <shogun/machine/StochasticGBMachine.h>
#include <shogun/mathematics/RandomNamespace.h>
#include <shogun/multiclass/tree/CARTree.h>

using namespace shogun;

class StochasticGBMachineTest : public ::testing::Test
{
protected:
	const int32_t num_train_samples = 100;
	const int32_t num_test_samples = 10;
	const int32_t dim = 1;
	const float64_t epsilon = 1e-8;
	std::mt19937_64 prng;

	void load_sinusoid_samples()
	{
		SGMatrix<float64_t> mat(dim, num_train_samples);
		random::fill_array(
		    mat.matrix, mat.matrix + num_train_samples, 0.0, 15.0, prng);

		SGVector<float64_t> labels(num_train_samples);
		for (int32_t i = 0; i < num_train_samples; i++)
			labels[i] = std::sin(mat(0, i));

		train_feats = std::make_shared<DenseFeatures<float64_t>>(mat);
		train_labels = std::make_shared<RegressionLabels>(labels);
	}

	void load_test_data()
	{
		SGVector<float64_t> tlab(num_test_samples);
		SGMatrix<float64_t> tdata(dim, num_test_samples);

		tlab[0] = -0.999585752311506259;
		tlab[1] = 0.75965469336929492;
		tlab[2] = -0.425832103506531334;
		tlab[3] = 0.298135616000050285;
		tlab[4] = -0.48828775732556795;
		tlab[5] = -0.031677813420380535;
		tlab[6] = 0.144672857935527394;
		tlab[7] = -0.0810247683026898424;
		tlab[8] = -0.767723534099077121;
		tlab[9] = 0.639868456911451666;

		tdata(0, 0) = 10.9667896982205075;
		tdata(0, 1) = 0.862781976084872615;
		tdata(0, 2) = 12.1264892751645501;
		tdata(0, 3) = 9.12203911322216499;
		tdata(0, 4) = 9.93490458930258313;
		tdata(0, 5) = 6.25150219333625934;
		tdata(0, 6) = 0.145182344164974608;
		tdata(0, 7) = 3.22270633960671393;
		tdata(0, 8) = 11.6910897047936668;
		tdata(0, 9) = 2.44726557225158103;

		test_feats = std::make_shared<DenseFeatures<float64_t>>(tdata);
		test_labels = std::make_shared<RegressionLabels>(tlab);
	}

	virtual void SetUp()
	{
		prng.seed(835);
		load_sinusoid_samples();
		load_test_data();
	}

public:
	std::shared_ptr<DenseFeatures<float64_t>> train_feats;
	std::shared_ptr<DenseFeatures<float64_t>> test_feats;
	std::shared_ptr<RegressionLabels> train_labels;
	std::shared_ptr<RegressionLabels> test_labels;

	StochasticGBMachineTest()
	    : train_feats(nullptr),
	      test_feats(nullptr),
	      train_labels(nullptr),
	      test_labels(nullptr)
	{
	}
};

#include <iostream>
TEST_F(StochasticGBMachineTest, sinusoid_curve_fitting)
{
	const int32_t seed = 2855;

	SGVector<bool> ft(1);
	ft[0]=false;
	auto tree=std::make_shared<CARTree>(ft);
	tree->set_max_depth(2);
	auto sq=std::make_shared<SquaredLoss>();
	auto sgbm = std::make_shared<StochasticGBMachine>(tree, sq, 100, 0.1, 1.0);
	sgbm->put("seed", seed);
	sgbm->set_labels(train_labels);
	sgbm->train(train_feats);

	auto ret_labels = sgbm->apply_regression(test_feats);
	SGVector<float64_t> ret=ret_labels->get_labels();

	EXPECT_NEAR(ret[0],-0.900765958,epsilon);
	EXPECT_NEAR(ret[1],0.7853974107,epsilon);
	EXPECT_NEAR(ret[2],-0.3759740238,epsilon);
	EXPECT_NEAR(ret[3],0.124319836,epsilon);
	EXPECT_NEAR(ret[4],-0.7288767951,epsilon);
	EXPECT_NEAR(ret[5],0.06000890273,epsilon);
	EXPECT_NEAR(ret[6],0.6541251248,epsilon);
	EXPECT_NEAR(ret[7],-0.09211134041,epsilon);
	EXPECT_NEAR(ret[8],-0.617577998,epsilon);
	EXPECT_NEAR(ret[9],0.5959804888,epsilon);
}

TEST_F(StochasticGBMachineTest, sinusoid_curve_fitting_subset_fraction)
{
	const int32_t seed = 2855;
	const float64_t fraction = 0.6;

	SGVector<bool> ft(1);
	ft[0] = false;
	auto tree = std::make_shared<CARTree>(ft);
	tree->set_max_depth(2);
	auto sq = std::make_shared<SquaredLoss>();

	auto sgbm = std::make_shared<StochasticGBMachine>(tree, sq, 100, 0.1, fraction);
	sgbm->put("seed", seed);
	sgbm->set_labels(train_labels);
	sgbm->train(train_feats);

	auto ret_labels = sgbm->apply_regression(test_feats);
	SGVector<float64_t> ret = ret_labels->get_labels();

	EXPECT_NEAR(ret[0], -0.5489844395, epsilon);
	EXPECT_NEAR(ret[1], 0.6958692923, epsilon);
	EXPECT_NEAR(ret[2], -0.299800903, epsilon);
	EXPECT_NEAR(ret[3], 0.1621379041, epsilon);
	EXPECT_NEAR(ret[4], -0.5489844395, epsilon);
	EXPECT_NEAR(ret[5], 0.06062667309, epsilon);
	EXPECT_NEAR(ret[6], 0.612491047, epsilon);
	EXPECT_NEAR(ret[7], -0.131144988, epsilon);
	EXPECT_NEAR(ret[8], -0.4408978052, epsilon);
	EXPECT_NEAR(ret[9], 0.5380825978, epsilon);
}
