/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;

void test()
{
	/*
	SGVector<float64_t> data(10);
	SGVector<float64_t>::range_fill_vector(data.vector, data.vlen, 1.0);

	float64_t low, up, mean;
	float64_t error_prob=0.05;
	mean=CStatistics::confidence_intervals_mean(data, error_prob, low, up);

	SG_SPRINT("sample mean: %f. True mean lies in [%f,%f] with %f%%\n",
			mean, low, up, 100*(1-error_prob));
	*/
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	test();

	exit_shogun();

	return 0;
}

