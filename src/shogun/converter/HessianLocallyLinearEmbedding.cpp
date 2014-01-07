/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Berlin Institute of Technology and Max-Planck-Society
 */

#include <converter/HessianLocallyLinearEmbedding.h>
#ifdef HAVE_EIGEN3
#include <kernel/LinearKernel.h>
#include <io/SGIO.h>
#include <lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

CHessianLocallyLinearEmbedding::CHessianLocallyLinearEmbedding() :
		CLocallyLinearEmbedding()
{
}

CHessianLocallyLinearEmbedding::~CHessianLocallyLinearEmbedding()
{
}

const char* CHessianLocallyLinearEmbedding::get_name() const
{
	return "HessianLocallyLinearEmbedding";
};

CFeatures* CHessianLocallyLinearEmbedding::apply(CFeatures* features)
{
	CKernel* kernel = new CLinearKernel((CDotFeatures*)features,(CDotFeatures*)features);
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.eigenshift = m_nullspace_shift;
	parameters.method = SHOGUN_HESSIAN_LOCALLY_LINEAR_EMBEDDING;
	parameters.target_dimension = m_target_dim;
	parameters.kernel = kernel;
	CDenseFeatures<float64_t>* embedding = tapkee_embed(parameters);
	SG_UNREF(kernel);
	return embedding;
}

#endif /* HAVE_EIGEN3 */
