/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/DenseFeatures.h>
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

