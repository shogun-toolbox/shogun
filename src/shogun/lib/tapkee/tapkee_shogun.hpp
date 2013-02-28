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
};

struct TAPKEE_PARAMETERS_FOR_SHOGUN
{
	TAPKEE_METHODS_FOR_SHOGUN method;
	uint32_t n_neighbors;
	uint32_t n_timesteps;
	uint32_t target_dimension;
	uint32_t spe_num_updates;
	float64_t eigenshift;
	float64_t landmark_ratio;
	float64_t gaussian_kernel_width;
	float64_t spe_tolerance;
	CKernel* kernel;
	CDistance* distance;
	CDotFeatures* features;
	bool spe_global_strategy;
	uint32_t max_iteration;
	float64_t fa_epsilon;
};

CDenseFeatures<float64_t>* tapkee_embed(const TAPKEE_PARAMETERS_FOR_SHOGUN& parameters);
}

#endif
#endif 
