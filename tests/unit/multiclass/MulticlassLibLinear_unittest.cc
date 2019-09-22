/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */
#include <gtest/gtest.h>
#include <shogun/base/ShogunEnv.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/NormalDistribution.h>

using namespace shogun;

TEST(MulticlassLibLinearTest,train_and_apply)
{
	int32_t seed = 100;
	index_t num_vec=10;
	index_t num_feat=3;
	index_t num_class=num_feat; // to make data easy
	float64_t distance=15;

	// create some linearly seperable data
	SGMatrix<float64_t> matrix(num_class, num_vec);
	SGMatrix<float64_t> matrix_test(num_class, num_vec);
	auto labels=std::make_shared<MulticlassLabels>(num_vec);
	auto labels_test=std::make_shared<MulticlassLabels>(num_vec);
	std::mt19937_64 prng(seed);
	NormalDistribution<float64_t> normal_dist;
	for (index_t i=0; i<num_vec; ++i)
	{
		index_t label=i%num_class;
		for (index_t j=0; j<num_feat; ++j)
		{
			matrix(j, i)=normal_dist(prng);
			matrix_test(j, i)=normal_dist(prng);
			labels->set_label(i, label);
			labels_test->set_label(i, label);
		}

		/* make sure data is linearly seperable per class */
		matrix(label, i)+=distance;
		matrix_test(label, i)+=distance;
	}
	//matrix.display_matrix("matrix");
	//labels->get_int_labels().display_vector("labels");

	// shogun will now own the matrix created
	auto features=std::make_shared<DenseFeatures<float64_t>>(matrix);
	auto features_test=std::make_shared<DenseFeatures<float64_t>>(
			matrix_test);


	float64_t C=1.0;

	auto mocas=std::make_shared<MulticlassLibLinear>(C, features,
			labels);
	env()->set_num_threads(1);
	mocas->set_epsilon(1e-5);
	mocas->train();

	auto pred=mocas->apply(features_test)->as<MulticlassLabels>();
	for (int i=0; i<features_test->get_num_vectors(); ++i)
		EXPECT_EQ(labels_test->get_label(i), pred->get_label(i));




}
