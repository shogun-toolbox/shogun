/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias Garcia
 */

#include <shogun/metric/LMNN.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <gtest/gtest.h>

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
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(feat_mat);

	// create labels
	SGVector<float64_t> lab_vec(4);
	lab_vec[0]=0;
	lab_vec[1]=0;
	lab_vec[2]=1;
	lab_vec[3]=1;
	// two-class data, use MulticlassLabels because it works in general for more than
	// two classes
	CMulticlassLabels* labels=new CMulticlassLabels(lab_vec);

	// create LMNN metric machine
	int32_t k=1;	// number of target neighbors per example
	CLMNN* lmnn=new CLMNN(features,labels,k);
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

	SG_UNREF(lmnn)
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
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(feat_mat);

	// create labels
	SGVector<float64_t> lab_vec(4);
	lab_vec[0]=0;
	lab_vec[1]=0;
	lab_vec[2]=1;
	lab_vec[3]=1;
	// two-class data, use MulticlassLabels because it works in general for more than
	// two classes
	CMulticlassLabels* labels=new CMulticlassLabels(lab_vec);

	// create LMNN metric machine
	int32_t k=1;	// number of target neighbors per example
	CLMNN* lmnn=new CLMNN(features,labels,k);
	// set number of maximum iterations and train
	lmnn->set_maxiter(500);
	lmnn->train();

	// check linear transform solution
	SGMatrix<float64_t> L=lmnn->get_linear_transform();
	EXPECT_NEAR(L(0,0),0.991577280560543,1e-5);
	EXPECT_NEAR(L(0,1),0,1e-5);
	EXPECT_NEAR(L(1,0),0,1e-5);
	EXPECT_NEAR(L(1,1),1.00000080000000002,1e-5);

	SG_UNREF(lmnn)
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
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(feat_mat);

	// create labels
	SGVector<float64_t> lab_vec(4);
	lab_vec[0]=0;
	lab_vec[1]=0;
	lab_vec[2]=1;
	lab_vec[3]=1;
	// two-class data, use MulticlassLabels because it works in general for more than
	// two classes
	CMulticlassLabels* labels=new CMulticlassLabels(lab_vec);

	// create LMNN metric machine
	int32_t k=1;	// number of target neighbors per example
	CLMNN* lmnn=new CLMNN(features,labels,k);
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

	SG_UNREF(lmnn)
}
