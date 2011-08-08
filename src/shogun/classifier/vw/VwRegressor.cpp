/*
 * Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
 * embodied in the content of this file are licensed under the BSD
 * (revised) open source license.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#include <shogun/classifier/vw/VwRegressor.h>
#include <shogun/loss/SquaredLoss.h>

using namespace shogun;

CVwRegressor::CVwRegressor()
{
	weight_vectors = NULL;
	loss = new CSquaredLoss();
	init(NULL);
}

CVwRegressor::CVwRegressor(CVwEnvironment* env)
{
	weight_vectors = NULL;
	loss = new CSquaredLoss();
	init(env);
}

CVwRegressor::~CVwRegressor()
{
	SG_FREE(weight_vectors);
	SG_UNREF(loss);
}

void CVwRegressor::init(CVwEnvironment* env)
{
	// For each feature, there should be 'stride' number of
	// elements in the weight vector
	index_t length = ((index_t) 1) << env->num_bits;
	env->thread_mask = (env->stride * (length >> env->thread_bits)) - 1;

	// Only one learning thread for now
	index_t num_threads = 1;
	weight_vectors = SG_MALLOC(float32_t*, num_threads);

	for (index_t i = 0; i < num_threads; i++)
	{
		weight_vectors[i] = SG_CALLOC(float, env->stride * length / num_threads);

		if (env->random_weights)
		{
			for (index_t j = 0; j < length/num_threads; j++)
				weight_vectors[i][j] = drand48() - 0.5;
		}

		if (env->initial_weight != 0.)
			for (index_t j = 0; j < env->stride*length/num_threads; j+=env->stride)
				weight_vectors[i][j] = env->initial_weight;

		if (env->adaptive)
			for (index_t j = 1; j < env->stride*length/num_threads; j+=env->stride)
				weight_vectors[i][j] = 1;
	}
}

