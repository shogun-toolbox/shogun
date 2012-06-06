/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 * Written (W) 2012 Victor Sadkov
 * Copyright (C) 2011 Moscow State University
 */

#include <shogun/base/init.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

void test_confidence_intervals()
{
	int32_t data_size=100;
	SGVector<float64_t> data(data_size);

	CMath::random_vector(data.vector, data.vlen, 0.0, 1.0);

	float64_t low, up, mean;
	float64_t error_prob=0.1;
	mean=CStatistics::confidence_intervals_mean(data, error_prob, low, up);

	SG_SPRINT("sample mean: %f. True mean lies in [%f,%f] with %f%%\n",
			mean, low, up, 100*(1-error_prob));

	SG_SPRINT("variance: %f\n", CStatistics::variance(data));
	SG_SPRINT("deviation: %f\n", CStatistics::std_deviation(data));
}

void test_inverse_incomplete_gamma()
{
	/* some tests for high precision MATLAB comparison */
	float64_t difference=CStatistics::inverse_incomplete_gamma(1, 1-0.95)*2;
	difference-=5.991464547107981;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-15);

	difference=CStatistics::inverse_incomplete_gamma(0.3, 1-0.95)*3;
	difference-=4.117049832302619;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-14)

	difference=CStatistics::inverse_incomplete_gamma(2, 1-0.95)*0.1;
	difference-=0.474386451839058;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-15)
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	test_confidence_intervals();
	test_inverse_incomplete_gamma();

	exit_shogun();

	return 0;
}

