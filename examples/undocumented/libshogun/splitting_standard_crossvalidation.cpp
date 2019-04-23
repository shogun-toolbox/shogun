/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/labels/RegressionLabels.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	index_t num_labels;
	index_t num_subsets;
	index_t runs=100;

	while (runs-->0)
	{
		num_labels=Math::random(10, 150);
		num_subsets=Math::random(1, 5);
		index_t desired_size=Math::round(
				(float64_t)num_labels/(float64_t)num_subsets);

		/* this will throw an error */
		if (num_labels<num_subsets)
			continue;

		SG_SPRINT("num_labels=%d\nnum_subsets=%d\n\n", num_labels, num_subsets);

		/* build labels */
		RegressionLabels* labels=new RegressionLabels(num_labels);
		for (index_t i=0; i<num_labels; ++i)
		{
			labels->set_label(i, Math::random(-10.0, 10.0));
			SG_SPRINT("label(%d)=%.18g\n", i, labels->get_label(i));
		}
		SG_SPRINT("\n");

		/* build splitting strategy */
		CrossValidationSplitting* splitting=
				new CrossValidationSplitting(labels, num_subsets);

		/* build index sets (twice to ensure memory is not leaking) */
		splitting->build_subsets();
		splitting->build_subsets();

		for (index_t i=0; i<num_subsets; ++i)
		{
			SG_SPRINT("subset %d\n", i);

			SGVector<index_t> subset=splitting->generate_subset_indices(i);
			SGVector<index_t> inverse=splitting->generate_subset_inverse(i);

			SGVector<index_t>::display_vector(subset.vector, subset.vlen, "subset indices");
			SGVector<index_t>::display_vector(inverse.vector, inverse.vlen, "inverse indices");

			SG_SPRINT("checking subset size: %d vs subset desired size %d\n",
					subset.vlen, desired_size);

			ASSERT(Math::abs(subset.vlen-desired_size)<=1);
			ASSERT(subset.vlen+inverse.vlen==num_labels);

			for (index_t j=0; j<subset.vlen; ++j)
				SG_SPRINT("%d:(%f),", subset.vector[j], labels->get_label(j));
			SG_SPRINT("\n");

			SG_SPRINT("inverse %d\n", i);
			for (index_t j=0; j<inverse.vlen; ++j)
				SG_SPRINT("%d(%d),", inverse.vector[j],
						(int32_t)labels->get_label(j));
			SG_SPRINT("\n\n");
		}

		/* clean up */
	}

	exit_shogun();

	return 0;
}

