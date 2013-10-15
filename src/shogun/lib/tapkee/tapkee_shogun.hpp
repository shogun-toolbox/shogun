/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_SHOGUN_ADAPTER
#define TAPKEE_SHOGUN_ADAPTER

#ifdef HAVE_EIGEN3

#include <shogun/io/SGIO.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

namespace shogun
{

enum TAPKEE_METHODS_FOR_SHOGUN
{
	SHOGUN_KERNEL_LOCALLY_LINEAR_EMBEDDING,
	SHOGUN_LOCALLY_LINEAR_EMBEDDING,
	SHOGUN_NEIGHBORHOOD_PRESERVING_EMBEDDING,
	SHOGUN_LOCAL_TANGENT_SPACE_ALIGNMENT,
	SHOGUN_LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT,
	SHOGUN_HESSIAN_LOCALLY_LINEAR_EMBEDDING,
	SHOGUN_DIFFUSION_MAPS,
	SHOGUN_LAPLACIAN_EIGENMAPS,
	SHOGUN_LOCALITY_PRESERVING_PROJECTIONS,
	SHOGUN_MULTIDIMENSIONAL_SCALING,
	SHOGUN_LANDMARK_MULTIDIMENSIONAL_SCALING,
	SHOGUN_ISOMAP,
	SHOGUN_LANDMARK_ISOMAP,
	SHOGUN_STOCHASTIC_PROXIMITY_EMBEDDING,
	SHOGUN_FACTOR_ANALYSIS,
	SHOGUN_TDISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING,
	SHOGUN_MANIFOLD_SCULPTING
};

struct TAPKEE_PARAMETERS_FOR_SHOGUN
{
	TAPKEE_PARAMETERS_FOR_SHOGUN() :
		method(SHOGUN_KERNEL_LOCALLY_LINEAR_EMBEDDING),
		n_neighbors(10), n_timesteps(3),
		target_dimension(2), spe_num_updates(100),
		eigenshift(1e-9), landmark_ratio(0.5),
		gaussian_kernel_width(1.0), spe_tolerance(1e-5),
		spe_global_strategy(false), max_iteration(100),
		fa_epsilon(1e-5), sne_theta(0.5),
		sne_perplexity(30.0), squishing_rate(0.99),
		kernel(NULL), distance(NULL), features(NULL)
	{
	}
	TAPKEE_METHODS_FOR_SHOGUN method;
	uint32_t n_neighbors;
	uint32_t n_timesteps;
	uint32_t target_dimension;
	uint32_t spe_num_updates;
	float64_t eigenshift;
	float64_t landmark_ratio;
	float64_t gaussian_kernel_width;
	float64_t spe_tolerance;
	bool spe_global_strategy;
	uint32_t max_iteration;
	float64_t fa_epsilon;
	float64_t sne_theta;
	float64_t sne_perplexity;
	float64_t squishing_rate;
	CKernel* kernel;
	CDistance* distance;
	CDotFeatures* features;
};

CDenseFeatures<float64_t>* tapkee_embed(const TAPKEE_PARAMETERS_FOR_SHOGUN& parameters);
}

#endif
#endif
