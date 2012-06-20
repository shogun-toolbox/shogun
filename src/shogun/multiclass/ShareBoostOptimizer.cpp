/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/optimization/lbfgs/lbfgs.h>
#include <shogun/multiclass/ShareBoostOptimizer.h>

using namespace shogun;

void ShareBoostOptimizer::optimize()
{
	int32_t N = m_sb->m_multiclass_strategy->get_num_classes() * m_sb->m_fea.num_rows;
	float64_t *W = lbfgs_malloc(N); // should use this function, if sse is enabled for liblbfgs
	lbfgs_parameter_t param;
	lbfgs_parameter_init(&param);

	for (int32_t i=0; i < N; ++I)
		W[i] = 0;


	lbfgs(N, W, &ShareBoostOptimizer::lbfgs_evaluate, NULL, this, &param);
	// TODO: assign W to each machines in m_sb
}

float64_t ShareBoostOptimizer::lbfgs_evaluate(void *userdata, float64_t *W, 
		float64_t *grad, const int32_t n, const float64_t step)
{
}
