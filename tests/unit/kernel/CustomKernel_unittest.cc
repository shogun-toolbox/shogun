/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/IndexFeatures.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(CustomKernelTest,add_row_subset)
{
	index_t m=3;
	CMeanShiftDataGenerator* gen=new CMeanShiftDataGenerator(0, 2);
	CFeatures* feats=gen->get_streamed_features(m);
	SG_REF(feats);

	CGaussianKernel* gauss=new CGaussianKernel(10, 3);
	gauss->init(feats, feats);
	CCustomKernel* custom=new CCustomKernel(gauss);

	SGVector<index_t> inds(m);
	inds.range_fill();

	index_t num_runs=10;
	for (index_t i=0; i<num_runs; ++i)
	{
		inds.permute();

		feats->add_subset(inds);
		custom->add_row_subset(inds);
		custom->add_col_subset(inds);
		gauss->init(feats, feats); // to make sure digonal is fine

		SGMatrix<float64_t> gauss_matrix=gauss->get_kernel_matrix();
		SGMatrix<float64_t> custom_matrix=custom->get_kernel_matrix();
//		gauss_matrix.display_matrix("gauss");
//		gauss_matrix.display_matrix("custom");
		for (index_t j=0; j<m*m; ++j)
			EXPECT_LE(CMath::abs(gauss_matrix.matrix[j]-custom_matrix.matrix[j]), 1E-6);

		feats->remove_subset();
		custom->remove_row_subset();
		custom->remove_col_subset();
	}

	SG_UNREF(gen);
	SG_UNREF(feats);
	SG_UNREF(gauss);
	SG_UNREF(custom);
}

TEST(CustomKernelTest,add_row_subset_constructor)
{
	index_t n=4;
	CMeanShiftDataGenerator* gen=new CMeanShiftDataGenerator(1, 2, 0);
	CDenseFeatures<float64_t>* feats=
			(CDenseFeatures<float64_t>*)gen->get_streamed_features(n);
	CGaussianKernel* gaussian=new CGaussianKernel(feats, feats, 2, 10);
	CCustomKernel* main_kernel=new CCustomKernel(gaussian);

	/* create custom kernel copy of gaussien and assert equalness */
	SGMatrix<float64_t> kmg=gaussian->get_kernel_matrix();
	SGMatrix<float64_t> km=main_kernel->get_kernel_matrix();
	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			EXPECT_LE(CMath::abs(kmg(i, j)-km(i, j)), 1E-7);
	}

	/* create copy of custom kernel and assert equalness */
	CCustomKernel* copy=new CCustomKernel(km);
	SGMatrix<float64_t> kmc=copy->get_kernel_matrix();
	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			EXPECT_EQ(km(i, j), kmc(i, j));
	}

	/* add a subset to the custom kernel, create copy, create another kernel
	 * from this, assert equalness */
	SGVector<index_t> inds(n);
	inds.range_fill();
	inds.permute();
	main_kernel->add_row_subset(inds);
	SGMatrix<float64_t> main_subset_matrix=main_kernel->get_kernel_matrix();
	main_kernel->remove_row_subset();
	CCustomKernel* main_subset_copy=new CCustomKernel(main_subset_matrix);
	SGMatrix<float64_t> main_subset_copy_matrix=main_subset_copy->get_kernel_matrix();
	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			EXPECT_EQ(main_subset_matrix(i, j), main_subset_copy_matrix(i, j));
	}

	SG_UNREF(main_subset_copy);
	SG_UNREF(gaussian);
	SG_UNREF(main_kernel);
	SG_UNREF(copy);
	SG_UNREF(gen);
}

TEST(CustomKernelTest,index_features_subset)
{
	index_t n=4;
	CMeanShiftDataGenerator* gen=new CMeanShiftDataGenerator(1, 2, 0);
	SG_REF(gen);
	CDenseFeatures<float64_t>* feats=
			(CDenseFeatures<float64_t>*)gen->get_streamed_features(n);
	SG_REF(feats);
	CGaussianKernel* gaussian=new CGaussianKernel(feats, feats, 2, 10);
	SG_REF(gaussian);
	CCustomKernel* main_kernel=new CCustomKernel(gaussian);
	SG_REF(main_kernel);

	/* create custom kernel copy of gaussien and assert equalness */
	SGMatrix<float64_t> kmg=gaussian->get_kernel_matrix();
	SGMatrix<float64_t> km=main_kernel->get_kernel_matrix();

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			EXPECT_LE(CMath::abs(kmg(i, j)-km(i, j)), 1E-7);
	}

	/* add a subset to the custom kernel, create copy, create another kernel
	 * from this, assert equalness */
	SGVector<index_t> r_idx(n);
	SGVector<index_t> c_idx(n);
	r_idx.range_fill();
	r_idx.permute();
	c_idx.range_fill();
	c_idx.permute();

	/* Create IndexFeatures instances */
	CIndexFeatures * feat_r_idx = new CIndexFeatures(r_idx);
	CIndexFeatures * feat_c_idx = new CIndexFeatures(c_idx);
	SG_REF(feat_r_idx);
	SG_REF(feat_c_idx);

	main_kernel->init(feat_r_idx,feat_c_idx);
	SGMatrix<float64_t> main_subset_matrix = main_kernel->get_kernel_matrix();

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			EXPECT_LE(CMath::abs(main_subset_matrix(i, j)-kmg(r_idx[i], c_idx[j])), 1e-7);
	}

	SG_UNREF(gaussian);
	SG_UNREF(main_kernel);
	SG_UNREF(gen);
	SG_UNREF(feats);
	SG_UNREF(feat_r_idx);
	SG_UNREF(feat_c_idx);
}
