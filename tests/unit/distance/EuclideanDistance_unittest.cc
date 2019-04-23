/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre, Soeren Sonnenburg, Fernando Iglesias,
 *          Soumyajit De, Bjoern Esser
 */

#include <gtest/gtest.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DenseSubSamplesFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/base/init.h>

using namespace shogun;

std::shared_ptr<DenseFeatures<float64_t>> create_lhs()
{
	// create object with two feature vectors,
	// each column represents a feature vector
	SGMatrix<float64_t> feat_mat_lhs(2,2);
	// 1st feature vector
	feat_mat_lhs(0,0)=0;
	feat_mat_lhs(1,0)=0;
	// 2nd feature vector
	feat_mat_lhs(0,1)=0;
	feat_mat_lhs(1,1)=-1;
	return std::make_shared<DenseFeatures<float64_t>>(feat_mat_lhs);
}

std::shared_ptr<DenseFeatures<float64_t>> create_rhs()
{
	// create object with two feature vectors,
	// each column represents a feature vector
	SGMatrix<float64_t> feat_mat_rhs(2,2);
	// 3rd feature vector
	feat_mat_rhs(0,0)=1;
	feat_mat_rhs(1,0)=1;
	// 4th feature vector
	feat_mat_rhs(0,1)=-1;
	feat_mat_rhs(1,1)=1;

	return std::make_shared<DenseFeatures<float64_t>>(feat_mat_rhs);
}

TEST(EuclideanDistance,distance)
{
	auto features_lhs=create_lhs();
	auto features_rhs=create_rhs();

	// put features into distance object to compute squared Euclidean distances
	auto euclidean=std::make_shared<EuclideanDistance>(features_lhs,features_rhs);
	euclidean->set_disable_sqrt(true);

	// check distances computed one by one
	EXPECT_EQ(euclidean->distance(0,0), 2);
	EXPECT_EQ(euclidean->distance(0,1), 2);
	EXPECT_EQ(euclidean->distance(1,0), 5);
	EXPECT_EQ(euclidean->distance(1,1), 5);


}

TEST(EuclideanDistance, distance_precomputed_norms)
{
	auto features_lhs=create_lhs();
	auto features_rhs=create_rhs();

	auto euclidean=std::make_shared<EuclideanDistance>(features_lhs,features_rhs);
	euclidean->set_disable_sqrt(true);
	euclidean->precompute_lhs();
	euclidean->precompute_rhs();

	// check distances computed one by one
	EXPECT_EQ(euclidean->distance(0,0), 2);
	EXPECT_EQ(euclidean->distance(0,1), 2);
	EXPECT_EQ(euclidean->distance(1,0), 5);
	EXPECT_EQ(euclidean->distance(1,1), 5);

	euclidean->reset_precompute();


}

TEST(EuclideanDistance,get_distance_matrix)
{
	auto features_lhs=create_lhs();
	auto features_rhs=create_rhs();

	// put features into distance object to compute squared Euclidean distances
	auto euclidean=std::make_shared<EuclideanDistance>(features_lhs,features_rhs);
	euclidean->set_disable_sqrt(true);
	euclidean->parallel->set_num_threads(1);

	SGMatrix<float64_t> distance_matrix=euclidean->get_distance_matrix();

	// check distance matrix
	EXPECT_EQ(distance_matrix(0,0), euclidean->distance(0,0));
	EXPECT_EQ(distance_matrix(0,1), euclidean->distance(0,1));
	EXPECT_EQ(distance_matrix(1,0), euclidean->distance(1,0));
	EXPECT_EQ(distance_matrix(1,1), euclidean->distance(1,1));


}

TEST(EuclideanDistance, heterogenous_features)
{
	auto features_lhs=create_lhs();
	auto features_rhs=create_rhs();



	SGVector<int32_t> idx(2);
	idx[0]=0;
	idx[1]=1;

	auto subsample_lhs=std::make_shared<DenseSubSamplesFeatures<float64_t>>(features_lhs, idx);
	auto subsample_rhs=std::make_shared<DenseSubSamplesFeatures<float64_t>>(features_rhs, idx);



	auto euclidean=std::make_shared<EuclideanDistance>();
	euclidean->set_disable_sqrt(true);
	euclidean->parallel->set_num_threads(1);

	float64_t accuracy=1E-15;

	euclidean->init(subsample_lhs, features_rhs);

	EXPECT_NEAR(euclidean->distance(0, 0), 2, accuracy);
	EXPECT_NEAR(euclidean->distance(0, 1), 2, accuracy);
	EXPECT_NEAR(euclidean->distance(1, 0), 5, accuracy);
	EXPECT_NEAR(euclidean->distance(1, 1), 5, accuracy);

	euclidean->init(features_lhs, subsample_rhs);

	EXPECT_NEAR(euclidean->distance(0, 0), 2, accuracy);
	EXPECT_NEAR(euclidean->distance(0, 1), 2, accuracy);
	EXPECT_NEAR(euclidean->distance(1, 0), 5, accuracy);
	EXPECT_NEAR(euclidean->distance(1, 1), 5, accuracy);






}
