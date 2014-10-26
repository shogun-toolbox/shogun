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
#include <shogun/mathematics/Math.h>

using namespace shogun;

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
		CMath::permute(subset);
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

	exit_shogun();
	return 0;
}


