/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <base/init.h>
#include <evaluation/StratifiedCrossValidationSplitting.h>
#include <labels/MulticlassLabels.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	index_t num_labels, num_classes, num_subsets;
	index_t runs=50;

	while (runs-->0)
	{
		num_labels=CMath::random(5, 100);
		num_classes=CMath::random(2, 10);
		num_subsets=CMath::random(1, 10);

		/* this will throw an error */
		if (num_labels<num_subsets)
			continue;

		SG_SPRINT("num_labels=%d\nnum_classes=%d\nnum_subsets=%d\n\n",
				num_labels, num_classes, num_subsets);

		/* build labels */
		CMulticlassLabels* labels=new CMulticlassLabels(num_labels);
		for (index_t i=0; i<num_labels; ++i)
		{
			labels->set_label(i, CMath::random()%num_classes);
			SG_SPRINT("label(%d)=%.18g\n", i, labels->get_label(i));
		}
		SG_SPRINT("\n");

		/* print classes */
		SGVector<float64_t> classes=labels->get_unique_labels();
		SGVector<float64_t>::display_vector(classes.vector, classes.vlen, "classes");

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

			SG_SPRINT("subset %d\n", i);
			for (index_t j=0; j<subset.vlen; ++j)
				SG_SPRINT("%d(%d),", subset.vector[j],
						(int32_t)labels->get_label(j));
			SG_SPRINT("\n");

			SG_SPRINT("inverse %d\n", i);
			for (index_t j=0; j<inverse.vlen; ++j)
				SG_SPRINT("%d(%d),", inverse.vector[j],
						(int32_t)labels->get_label(j));
			SG_SPRINT("\n\n");
		}

		/* check whether number of labels in every subset is nearly equal */
		for (index_t i=0; i<num_classes; ++i)
		{
			SG_SPRINT("checking class %d\n", i);

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
				SG_SPRINT("number in subset %d: %d\n", j, temp_count);
				ASSERT(CMath::abs(temp_count-count)<=1);
			}
		}

		/* clean up */
		SG_UNREF(splitting);
	}

	exit_shogun();

	return 0;
}

