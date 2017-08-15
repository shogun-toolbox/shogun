/*
 * Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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