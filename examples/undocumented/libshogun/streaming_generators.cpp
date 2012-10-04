/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 *
 * This file demonstrates how to use data generators based on the streaming
 * features framework
 */

#include <shogun/base/init.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>

using namespace shogun;

void test_mean_shift()
{
	index_t dimension=3;
	index_t mean_shift=100;
	index_t num_runs=1000;

	CMeanShiftDataGenerator<float64_t>* gen=
				new CMeanShiftDataGenerator<float64_t>(mean_shift, dimension);

	SGVector<float64_t> avg(dimension);
	avg.zero();

	for (index_t  i=0; i<num_runs; ++i)
	{
		gen->get_next_example();
		avg.add(gen->get_vector());
	}

	/* average */
	avg.scale(1.0/num_runs);
	avg.display_vector("mean_shift");

	/* roughly assert correct model parameters */
	ASSERT(avg[0]-mean_shift<mean_shift/100);
	for (index_t i=1; i<dimension; ++i)
		ASSERT(avg[i]<0.5 && avg[i]>-0.5);

	SG_UNREF(gen);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	test_mean_shift();

	exit_shogun();
	return 0;
}

