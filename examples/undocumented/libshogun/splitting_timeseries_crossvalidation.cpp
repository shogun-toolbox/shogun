/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2017 Sahil Chaddha
 */

#include <shogun/base/init.h>
#include <shogun/evaluation/TimeSeriesSplitting.h>
#include <shogun/labels/RegressionLabels.h>
using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char** argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	index_t num_labels;
	index_t num_subsets;
	index_t h_value;
	index_t runs = 100;

	while (runs-- > 0)
	{
		num_labels = CMath::random(10, 150);
		num_subsets = CMath::random(1, 5);
		h_value = CMath::random(1, 30);
		/* this will throw an error */
		if (num_labels < num_subsets)
			continue;

		SG_SPRINT(
		    "num_labels=%d\nnum_subsets=%d\nh_value=%d\n\n", num_labels,
		    num_subsets, h_value);

		/* build labels */
		CRegressionLabels* labels = new CRegressionLabels(num_labels);
		for (index_t i = 0; i < num_labels; ++i)
		{
			labels->set_label(i, CMath::random(-10.0, 10.0));
			SG_SPRINT("label(%d)=%.18g\n", i, labels->get_label(i));
		}
		SG_SPRINT("\n");

		/* build splitting strategy */
		CTimeSeriesSplitting* splitting =
		    new CTimeSeriesSplitting(labels, num_subsets);

		/* setting h_value */
		splitting->set_h(h_value);

		/* build index sets (twice to ensure memory is not leaking) */
		splitting->build_subsets();
		splitting->build_subsets();

		for (index_t i = 0; i < num_subsets; ++i)
		{
			SG_SPRINT("subset %d\n", i);

			SGVector<index_t> subset = splitting->generate_subset_indices(i);
			SGVector<index_t> inverse = splitting->generate_subset_inverse(i);

			SGVector<index_t>::display_vector(
			    subset.vector, subset.vlen, "subset indices");
			SGVector<index_t>::display_vector(
			    inverse.vector, inverse.vlen, "inverse indices");

			SG_SPRINT("subset size: %d\n", subset.vlen);

			ASSERT(CMath::abs(num_labels - subset.vlen) >= 1);
			ASSERT(subset.vlen + inverse.vlen == num_labels);
			/* Every test should have h future. */
			ASSERT(subset.vlen >= splitting->get_h());

			for (index_t j = 0; j < subset.vlen; ++j)
				SG_SPRINT("%d:(%f),", subset.vector[j], labels->get_label(j));
			SG_SPRINT("\n");

			SG_SPRINT("inverse %d\n", i);
			for (index_t j = 0; j < inverse.vlen; ++j)
				SG_SPRINT(
				    "%d(%d),", inverse.vector[j],
				    (int32_t)labels->get_label(j));
			SG_SPRINT("\n\n");
		}

		/* clean up */
		SG_UNREF(splitting);
	}

	exit_shogun();

	return 0;
}