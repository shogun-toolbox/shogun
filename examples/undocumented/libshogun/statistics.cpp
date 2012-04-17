/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 * Written (W) 2012 Victor Sadkov
 * Copyright (C) 2011 Moscow State University
 */

#include <shogun/base/init.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

const int DATA_SIZE=100;

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	SGVector<float64_t> data(DATA_SIZE, true);
	CMath::random_vector(data.vector, data.vlen, 0.0, 1.0);

	// for (int32_t i=0; i<DATA_SIZE; i++)
		// SG_SPRINT("data[%02d] = %.5lf%s", i, data[i], (i+1)%4?"\t":"\n");

	float64_t low, up, mean;
	float64_t error_prob=0.1;
	mean=CStatistics::confidence_intervals_mean(data, error_prob, low, up);

	SG_SPRINT("sample mean: %f. True mean lies in [%f,%f] with %f%%\n",
			mean, low, up, 100*(1-error_prob));

	SG_SPRINT("variance: %f\n", CStatistics::variance(data));
	SG_SPRINT("deviation: %f\n", CStatistics::std_deviation(data));

	data.free_vector();

	SG_SPRINT("\nEND\n");
	exit_shogun();

	return 0;
}

