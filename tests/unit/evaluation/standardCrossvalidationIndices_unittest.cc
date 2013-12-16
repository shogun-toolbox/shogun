/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Saurabh Mahindre
 */

#include <shogun/base/init.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/labels/RegressionLabels.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(standardCrossvalidationIndices,one)
{	
	index_t num_indices;
	index_t num_labels;
	index_t num_subsets;
	index_t runs=100;

	while (runs-->0)
	{	
		num_indices=0;
		num_labels=CMath::random(10, 150);
		num_subsets=CMath::random(1, 5);
		index_t desired_size=CMath::round(
				(float64_t)num_labels/(float64_t)num_subsets);

		/* this will throw an error */
		if (num_labels<num_subsets)
			continue;

		/* build labels */
		CRegressionLabels* labels=new CRegressionLabels(num_labels);
		for (index_t i=0; i<num_labels; ++i)
		{
			labels->set_label(i, CMath::random(-10.0, 10.0));
		}

		/* build splitting strategy */
		CCrossValidationSplitting* splitting=
				new CCrossValidationSplitting(labels, num_subsets);

		/* build index sets (twice to ensure memory is not leaking) */
		splitting->build_subsets();
		splitting->build_subsets();

		for (index_t i=0; i<num_subsets; ++i)
		{
			SGVector<index_t> subset=splitting->generate_subset_indices(i);
			SGVector<index_t> inverse=splitting->generate_subset_inverse(i);

			num_indices = num_indices + subset.vlen;

			EXPECT_LE(CMath::abs(subset.vlen-desired_size),1);
			EXPECT_EQ(subset.vlen+inverse.vlen,num_labels);
		}

		EXPECT_EQ(num_indices,num_labels);

		/* clean up */
		SG_UNREF(splitting);
	}

}

