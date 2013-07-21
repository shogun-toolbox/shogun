/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias Garcia
 */

#include <shogun/metric/LMNNImpl.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <gtest/gtest.h>

using namespace shogun;

#ifdef HAVE_EIGEN3

TEST(LMNNImpl,find_target_nn)
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
	// two-class data, use MulticlassLabels because LMNN works in general for more than
	// two classes
	CMulticlassLabels* labels=new CMulticlassLabels(lab_vec);

	// find target neighbors
	int32_t k=1;	// number of target neighbors per example
	SGMatrix<index_t> target_nn=CLMNNImpl::find_target_nn(features,labels,k);

	// check output dimensions
	EXPECT_EQ(target_nn.num_rows, k);
	EXPECT_EQ(target_nn.num_cols, features->get_num_vectors());
	// check output contents
	EXPECT_EQ(target_nn(0,0), 1);
	EXPECT_EQ(target_nn(0,1), 0);
	EXPECT_EQ(target_nn(0,2), 3);
	EXPECT_EQ(target_nn(0,3), 2);

	SG_UNREF(features)
	SG_UNREF(labels)
}

TEST(LMNNImpl,compute_outer_products)
{
	// create features, each column is a feature vector
	SGMatrix<float64_t> feat_mat(2,2);
	// 1st feature vector
	feat_mat(0,0)=1;
	feat_mat(1,0)=2;
	// 2nd feature vector
	feat_mat(0,1)=-3;
	feat_mat(1,1)=4;
	// wrap feat_mat in Shogun features
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(feat_mat);
	// shorthand for the number of features
	int32_t d=features->get_num_features();

	// compute the outer products of the differences between every two pairs of
	// feature vectors
	OuterProductsMatrixType C=CLMNNImpl::compute_outer_products(features);

	// check output dimensions
	EXPECT_EQ(C.size(), 2);
	for (uint32_t i=0; i<C.size(); i++)
		EXPECT_EQ(C[i].size(), 2);

	for (uint32_t i=0; i<C.size(); i++)
		for (uint32_t j=0; j<C[i].size(); j++)
		{
			EXPECT_EQ(C[i][j].rows(), d);
			EXPECT_EQ(C[i][j].cols(), d);
		}

	// check the outer products that must be zero
	for (int32_t i=0; i<C[0][0].rows(); i++)
		for (int32_t j=0; j<C[0][0].cols(); j++)
		{
			EXPECT_EQ(C[0][0](i,j), 0);
			EXPECT_EQ(C[1][1](i,j), 0);
		}

	// check that the other two outer products are equal...
	for (int32_t i=0; i<C[0][1].rows(); i++)
		for (int32_t j=0; j<C[0][1].cols(); j++)
		{
			EXPECT_EQ(C[0][1](i,j), C[1][0](i,j));
		}

	// ...and check their results
	EXPECT_EQ(C[0][1](0,0), 16);
	EXPECT_EQ(C[0][1](1,0), -8);
	EXPECT_EQ(C[0][1](0,1), -8);
	EXPECT_EQ(C[0][1](1,1), 4);

	SG_UNREF(features)
}

TEST(LMNNImpl,sum_outer_products)
{
	// fill in matrix of outer products with arbitrary matrices
	OuterProductsMatrixType op;
	int32_t size=2;
	op.resize(size);
	for (int32_t i=0; i<size; i++)
		op[i].resize(size);

	for (int32_t i=0; i<size; i++)
		for (int32_t j=0; j<size; j++)
			op[i][j]=Eigen::MatrixXd(2,2);

	op[0][0] << 1,2,3,4;
	op[0][1] << -1,7,3,-4;
	op[1][0] << 8,9,-5,2;
	op[1][1] << 3,2,7,4;

	// matrix of target neighbors
	SGMatrix<index_t> target_nn(1,2);
	target_nn(0,0)=1;
	target_nn(0,1)=0;

	Eigen::MatrixXd sop=CLMNNImpl::sum_outer_products(op,target_nn);
	Eigen::MatrixXd test=op[0][1]+op[1][0];

	for (int32_t i=0; i<size; i++)
		for (int32_t j=0; j<size; j++)
			EXPECT_EQ(sop(i,j), test(i,j));

	// try with another matrix of target neighbors
	target_nn=SGMatrix<index_t>(1,1);
	target_nn(0,0)=0;
	sop=CLMNNImpl::sum_outer_products(op,target_nn);

	for (int32_t i=0; i<size; i++)
		for (int32_t j=0; j<size; j++)
			EXPECT_EQ(sop(i,j), op[0][0](i,j));

	// input an empty matrix of target neighbors
	sop=CLMNNImpl::sum_outer_products(op,SGMatrix<index_t>(0,0));
	for (int32_t i=0; i<size; i++)
		for (int32_t j=0; j<size; j++)
			EXPECT_EQ(sop(i,j), 0);
}

TEST(LMNNImpl,find_impostors_exact)
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
	// shorthand for the number of features
	int32_t d=features->get_num_features();

	// create labels
	SGVector<float64_t> lab_vec(4);
	lab_vec[0]=0;
	lab_vec[1]=0;
	lab_vec[2]=1;
	lab_vec[3]=1;
	// two-class data, use MulticlassLabels because LMNN works in general for more than
	// two classes
	CMulticlassLabels* labels=new CMulticlassLabels(lab_vec);

	// find target neighbors
	int32_t k=1;	// number of target neighbors per example
	SGMatrix<index_t> target_nn=CLMNNImpl::find_target_nn(features,labels,k);

	// find impostors with exact search (force exact search by setting correction=1)
	ImpostorsSetType impostors=CLMNNImpl::find_impostors(features,labels,
			Eigen::MatrixXd::Identity(d,d),target_nn,0,1);

	// impostors ground truth computed externally
	index_t impostors_arr[] = {0,1,2, 0,1,3, 2,3,0, 2,3,1, 3,2,0, 3,2,1};
	// each column is an impostor triplet
	SGMatrix<index_t> impostors_gt(impostors_arr,3,6,false);

	// check the impostor triplets
	EXPECT_EQ(impostors.size(), impostors_gt.num_cols); // number of impostor triplets

	int32_t i=0;
	for (ImpostorsSetType::iterator it=impostors.begin(); it!=impostors.end(); it++,i++)
	{
		EXPECT_EQ(impostors_gt(0,i), it->example);
		EXPECT_EQ(impostors_gt(1,i), it->target);
		EXPECT_EQ(impostors_gt(2,i), it->impostor);
	}

	SG_UNREF(features)
	SG_UNREF(labels)
}

#endif /* HAVE_EIGEN3 */
