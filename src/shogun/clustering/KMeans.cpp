/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Parijat Mazumdar
 */

#include "shogun/clustering/KMeans.h"
#include <shogun/clustering/KMeansLloydImpl.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

namespace shogun
{
CKMeans::CKMeans() : CKMeansBase()
{
}

CKMeans::CKMeans(int32_t k, CDistance* d, bool use_kmpp) : CKMeansBase(k, d, use_kmpp)
{
}

CKMeans::CKMeans(int32_t k_i, CDistance* d_i, SGMatrix<float64_t> centers_i) : CKMeansBase(k_i, d_i, centers_i)
{
}

CKMeans::~CKMeans()
{
}

bool CKMeans::train_machine(CFeatures* data)
{
	int32_t XSize;
	XSize=initialize_training(data);	
	
	SGVector<int32_t> ClList=SGVector<int32_t>(XSize);
	ClList.zero();
	SGVector<float64_t> weights_set=SGVector<float64_t>(m_k);
	weights_set.zero();

	if (m_mus_initial.matrix)
		set_initial_centers(weights_set, ClList, XSize);
	else
		set_random_centers(weights_set, ClList, XSize);

	CKMeansLloydImpl::Lloyd_KMeans(m_k, distance, m_max_iter, m_mus, ClList, weights_set, m_fixed_centers);

	compute_cluster_variances();
	return true;
}

}
