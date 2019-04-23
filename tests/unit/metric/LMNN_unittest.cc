/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Soeren Sonnenburg, Pan Deng, Wuwei Lin,
 *          Viktor Gal
 */
#include <gtest/gtest.h>

#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/metric/LMNN.h>

using namespace shogun;

TEST(LMNN,train_identity_init)
{
	// create features, each column is a feature vector
	SGMatrix<float64_t> feat_mat(2,4);
	// 1st feature vector
	feat_mat(0,0)=0;
	feat_mat(1,0)=0;
	// 2nd feature vector
	feat_mat(0,1)=0;
	feat_mat(1,1)=-1;
	// 3rd feature vector
	feat_mat(0,2)=1;
	feat_mat(1,2)=1;
	// 4th feature vector
	feat_mat(0,3)=-1;
	feat_mat(1,3)=1;
	// wrap feat_mat in Shogun features
	auto features=std::make_shared<DenseFeatures<float64_t>>(feat_mat);

	// create labels
	SGVector<float64_t> lab_vec(4);
	lab_vec[0]=0;
	lab_vec[1]=0;
	lab_vec[2]=1;
	lab_vec[3]=1;
	// two-class data, use MulticlassLabels because it works in general for more than
	// two classes
	auto labels=std::make_shared<MulticlassLabels>(lab_vec);

	// create LMNN metric machine
	int32_t k=1;	// number of target neighbors per example
	auto lmnn=std::make_shared<LMNN>(features,labels,k);
	// use the identity matrix as initial transform for LMNN
	SGMatrix<float64_t> init_transform=SGMatrix<float64_t>::create_identity_matrix(2,1);
	// set number of maximum iterations and train
	lmnn->set_maxiter(500);
	lmnn->train(init_transform);

	// check linear transform solution
	SGMatrix<float64_t> L=lmnn->get_linear_transform();
	EXPECT_NEAR(L(0,0),0.991577280560543,1e-5);
	EXPECT_NEAR(L(0,1),0,1e-5);
	EXPECT_NEAR(L(1,0),0,1e-5);
	EXPECT_NEAR(L(1,1),1.00000080000000002,1e-5);


}

TEST(LMNN, train_termination)
{
	SGMatrix<float64_t> feat_mat(2, 4);
	feat_mat(0, 0) = 0;
	feat_mat(1, 0) = 0;
	feat_mat(0, 1) = 0;
	feat_mat(1, 1) = -1;
	feat_mat(0, 2) = 1;
	feat_mat(1, 2) = 1;
	feat_mat(0, 3) = -1;
	feat_mat(1, 3) = 1;

	auto features =
	    std::make_shared<DenseFeatures<float64_t>>(feat_mat);

	SGVector<float64_t> lab_vec(4);
	lab_vec[0] = 0;
	lab_vec[1] = 0;
	lab_vec[2] = 1;
	lab_vec[3] = 1;

	auto labels = std::make_shared<MulticlassLabels>(lab_vec);

	int32_t k = 1; // number of target neighbors per example
	auto lmnn = std::make_shared<LMNN>(features, labels, k);

	SGMatrix<float64_t> init_transform =
	    SGMatrix<float64_t>::create_identity_matrix(2, 1);
	lmnn->set_maxiter(1500);
	lmnn->train(init_transform);

	// check linear transform solution
	SGMatrix<float64_t> L = lmnn->get_linear_transform();
	EXPECT_NEAR(L(0, 0), 0.000041647483219, 1e-10);
	EXPECT_NEAR(L(0, 1), 0, 1e-10);
	EXPECT_NEAR(L(1, 0), 0, 1e-10);
	EXPECT_NEAR(L(1, 1), 0.988162395685451, 1e-10);

	// check number of iterations
	auto stat = lmnn->get_statistics();
	EXPECT_EQ(stat->obj.vlen, 1234);
}

TEST(LMNN,train_pca_init)
{
	// create features, each column is a feature vector
	SGMatrix<float64_t> feat_mat(2,4);
	// 1st feature vector
	feat_mat(0,0)=0;
	feat_mat(1,0)=0;
	// 2nd feature vector
	feat_mat(0,1)=0;
	feat_mat(1,1)=-1;
	// 3rd feature vector
	feat_mat(0,2)=1;
	feat_mat(1,2)=1;
	// 4th feature vector
	feat_mat(0,3)=-1;
	feat_mat(1,3)=1;
	// wrap feat_mat in Shogun features
	auto features=std::make_shared<DenseFeatures<float64_t>>(feat_mat);

	// create labels
	SGVector<float64_t> lab_vec(4);
	lab_vec[0]=0;
	lab_vec[1]=0;
	lab_vec[2]=1;
	lab_vec[3]=1;
	// two-class data, use MulticlassLabels because it works in general for more than
	// two classes
	auto labels=std::make_shared<MulticlassLabels>(lab_vec);

	// create LMNN metric machine
	int32_t k=1;	// number of target neighbors per example
	auto lmnn=std::make_shared<LMNN>(features,labels,k);
	// set number of maximum iterations and train
	lmnn->set_maxiter(500);
	lmnn->train();

	// check linear transform solution
	SGMatrix<float64_t> L=lmnn->get_linear_transform();
	EXPECT_NEAR(L(0,0),0.991577280560543,1e-5);
	EXPECT_NEAR(L(0,1),0,1e-5);
	EXPECT_NEAR(L(1,0),0,1e-5);
	EXPECT_NEAR(L(1,1),1.00000080000000002,1e-5);


}

TEST(LMNN,train_diagonal)
{
	// create features, each column is a feature vector
	SGMatrix<float64_t> feat_mat(2,4);
	// 1st feature vector
	feat_mat(0,0)=0;
	feat_mat(1,0)=0;
	// 2nd feature vector
	feat_mat(0,1)=0;
	feat_mat(1,1)=-1;
	// 3rd feature vector
	feat_mat(0,2)=1;
	feat_mat(1,2)=1;
	// 4th feature vector
	feat_mat(0,3)=-1;
	feat_mat(1,3)=1;
	// wrap feat_mat in Shogun features
	auto features=std::make_shared<DenseFeatures<float64_t>>(feat_mat);

	// create labels
	SGVector<float64_t> lab_vec(4);
	lab_vec[0]=0;
	lab_vec[1]=0;
	lab_vec[2]=1;
	lab_vec[3]=1;
	// two-class data, use MulticlassLabels because it works in general for more than
	// two classes
	auto labels=std::make_shared<MulticlassLabels>(lab_vec);

	// create LMNN metric machine
	int32_t k=1;	// number of target neighbors per example
	auto lmnn=std::make_shared<LMNN>(features,labels,k);
	// use the identity matrix as initial transform for LMNN
	SGMatrix<float64_t> init_transform=SGMatrix<float64_t>::create_identity_matrix(2,1);
	// set number of maximum iterations and train
	lmnn->set_maxiter(1000);
	lmnn->set_diagonal(true);
	lmnn->train(init_transform);

	// check linear transform solution
	SGMatrix<float64_t> L=lmnn->get_linear_transform();
	EXPECT_NEAR(L(0,0),0.61938,1e-5);
	EXPECT_NEAR(L(0,1),0,1e-5);
	EXPECT_NEAR(L(1,0),0,1e-5);
	EXPECT_NEAR(L(1,1),1.0000,1e-5);


}
