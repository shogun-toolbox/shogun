/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Saurabh Mahindre
 */

#include <shogun/base/init.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/labels/MulticlassLabels.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(stratifiedCrossvalidationIndices,one)
{
	index_t num_labels, num_classes, num_subsets, num_indices;
	index_t runs=50;

	while (runs-->0)
	{	
		num_indices=0;
		num_labels=CMath::random(5, 100);
		num_classes=CMath::random(2, 10);
		num_subsets=CMath::random(1, 10);

		/* this will throw an error */
		if (num_labels<num_subsets)
			continue;


		/* build labels */
		CMulticlassLabels* labels=new CMulticlassLabels(num_labels);
		for (index_t i=0; i<num_labels; ++i)
		{
			labels->set_label(i, CMath::random()%num_classes);
		}

		/* print classes */
		SGVector<float64_t> classes=labels->get_unique_labels();

		/* build splitting strategy */
		CStratifiedCrossValidationSplitting* splitting=
				new CStratifiedCrossValidationSplitting(labels, num_subsets);

		/* build index sets (twice to ensure memory is not leaking) */
		splitting->build_subsets();
		splitting->build_subsets();

		for (index_t i=0; i<num_subsets; ++i)
		{
			SGVector<index_t> subset=splitting->generate_subset_indices(i);
			SGVector<index_t> inverse=splitting->generate_subset_inverse(i);
			
			EXPECT_EQ(subset.vlen+inverse.vlen, num_labels);
			num_indces = num_indices + subset.vlen;
		}

		EXPECT_EQ(num_indices, num_labels);

		/* check whether number of labels in every subset is nearly equal */
		for (index_t i=0; i<num_classes; ++i)
		{
			/* count number of elements for this class */
			SGVector<index_t> temp=splitting->generate_subset_indices(0);
			int32_t count=0;
			for (index_t j=0; j<temp.vlen; ++j)
			{
				if ((int32_t)labels->get_label(temp.vector[j])==i)
					++count;
			}

			/* check all subsets for same ratio */
			for (index_t j=0; j<num_subsets; ++j)
			{
				SGVector<index_t> subset=splitting->generate_subset_indices(j);
				int32_t temp_count=0;
				for (index_t k=0; k<subset.vlen; ++k)
				{
					if ((int32_t)labels->get_label(subset.vector[k])==i)
						++temp_count;
				}

				/* at most one difference */
				EXPECT_LE(CMath::abs(temp_count-count),1);
			}
		}

		/* clean up */
		SG_UNREF(splitting);
	}
}


