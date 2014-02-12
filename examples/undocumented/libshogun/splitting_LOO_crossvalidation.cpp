/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Saurabh Mahindre
 */

#include <shogun/base/init.h>
#include <shogun/evaluation/LOOCrossValidationSplitting.h>
#include <shogun/labels/RegressionLabels.h>

using namespace shogun;

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	index_t num_labels;
	index_t runs=10;

	while (runs-->0)
	{
		num_labels=CMath::random(10, 50);

		//SG_SPRINT("num_labels=%d\n\n", num_labels);

		/* build labels */
		CRegressionLabels* labels=new CRegressionLabels(num_labels);
		for (index_t i=0; i<num_labels; ++i)
		{
			labels->set_label(i, CMath::random(-10.0, 10.0));
		//	SG_SPRINT("label(%d)=%.18g\n", i, labels->get_label(i));
		
		}
		
		//SG_SPRINT("\n");

		/* build Leave one out splitting strategy */
		CLOOCrossValidationSplitting* splitting=
				new CLOOCrossValidationSplitting(labels);

		splitting->build_subsets();

		for (index_t i=0; i<num_labels; ++i)
		{
			//SG_SPRINT("subset %d\n", i);

			SGVector<index_t> subset=splitting->generate_subset_indices(i);
			SGVector<index_t> inverse=splitting->generate_subset_inverse(i);

			SGVector<index_t>::display_vector(subset.vector, subset.vlen, "subset indices");
			SGVector<index_t>::display_vector(inverse.vector, inverse.vlen, "inverse indices");


			/*for (index_t j=0; j<subset.vlen; ++j)
				SG_SPRINT("%d:(%f),", subset.vector[j], labels->get_label(subset.vector[j]));
			SG_SPRINT("\n");

			SG_SPRINT("inverse %d\n", i);
			for (index_t j=0; j<inverse.vlen; ++j)
				SG_SPRINT("%d(%d),", inverse.vector[j],
						(int32_t)labels->get_label(inverse.vector[j]));
			SG_SPRINT("\n\n");
			*/
		}

		/* clean up */
		SG_UNREF(splitting);
	}

	exit_shogun();

	return 0;
}


