/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/IndexFeatures.h>

using namespace shogun;

void test_custom_kernel_index_subsets()
{
	/* create some data */
	index_t m=10;
	CFeatures* features=
			new CDenseFeatures<float64_t>(CDataGenerator::generate_mean_data(
			m, 2, 1));
	SG_REF(features);

	/* create a custom kernel */
	CGaussianKernel* gaussian_kernel=new CGaussianKernel(2,10);
	gaussian_kernel->init(features, features);
	CCustomKernel* custom_kernel=new CCustomKernel(gaussian_kernel);

	/* create random permutations */
	SGVector<index_t> row_subset(m);
	SGVector<index_t> col_subset(m);
	row_subset.range_fill();
	row_subset.permute();
	col_subset.range_fill();
	col_subset.permute();

	/* create index features */
	CIndexFeatures* row_idx_feat=new CIndexFeatures(row_subset);
	CIndexFeatures* col_idx_feat=new CIndexFeatures(col_subset);
	SG_REF(row_idx_feat);
	SG_REF(col_idx_feat);

	custom_kernel->init(row_idx_feat, col_idx_feat);

	SGMatrix<float64_t> gaussian_kernel_matrix=
			gaussian_kernel->get_kernel_matrix();

	SGMatrix<float64_t> custom_kernel_matrix=
			custom_kernel->get_kernel_matrix();

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<m; ++j)
		{
			SG_SDEBUG("Custom(%d,%d)=%f, Gaussian(%d,%d)=%f\n",
					i, j, custom_kernel_matrix(i, j),
					i, j, gaussian_kernel_matrix(row_subset[i], col_subset[j]));
			ASSERT(CMath::abs(custom_kernel_matrix(i, j)-
					gaussian_kernel_matrix(row_subset[i], col_subset[j]))<10E-8);
		}
	}

	SG_UNREF(gaussian_kernel);
	SG_UNREF(custom_kernel);
	SG_UNREF(row_idx_feat);
	SG_UNREF(col_idx_feat);
	SG_UNREF(features);
}

void test_custom_kernel_subsets()
{
	/* create some data */
	index_t m=10;
	CFeatures* features=
			new CDenseFeatures<float64_t>(CDataGenerator::generate_mean_data(
			m, 2, 1));
	SG_REF(features);

	/* create a custom kernel */
	CKernel* k=new CGaussianKernel();
	k->init(features, features);
	CCustomKernel* l=new CCustomKernel(k);

	/* create a random permutation */
	SGVector<index_t> subset(m);

	for (index_t run=0; run<100; ++run)
	{
		subset.range_fill();
		subset.permute();
//		subset.display_vector("permutation");
		features->add_subset(subset);
		k->init(features, features);
		l->add_row_subset(subset);
		l->add_col_subset(subset);
//		k->get_kernel_matrix().display_matrix("K");
//		l->get_kernel_matrix().display_matrix("L");
		for (index_t i=0; i<m; ++i)
		{
			for (index_t j=0; j<m; ++j)
			{
				SG_SDEBUG("K(%d,%d)=%f, L(%d,%d)=%f\n", i, j, k->kernel(i, j), i, j,
						l->kernel(i, j));
				ASSERT(CMath::abs(k->kernel(i, j)-l->kernel(i, j))<10E-8);
			}
		}

		features->remove_subset();
		l->remove_row_subset();
		l->remove_col_subset();
	}

	SG_UNREF(k);
	SG_UNREF(l);
	SG_UNREF(features);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	//sg_io->set_loglevel(MSG_DEBUG);

	test_custom_kernel_subsets();
	test_custom_kernel_index_subsets();

	exit_shogun();
	return 0;
}


