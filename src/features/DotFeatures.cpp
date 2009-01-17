/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/DotFeatures.h"
#include "lib/io.h"

void CDotFeatures::dense_dot_range(float64_t* output, int32_t start, int32_t stop, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b)
{
	ASSERT(output);
	ASSERT(start>=0);
	ASSERT(start<=stop);
	ASSERT(stop<=get_num_vectors());

	if (alphas)
	{
		for (int32_t i=start; i<stop; i++)
			output[i]=alphas[i]*dense_dot(i, vec, dim)+b;
	}
	else
	{
		for (int32_t i=start; i<stop; i++)
			output[i]=alphas[i]*dense_dot(i, vec, dim)+b;
	}
}
