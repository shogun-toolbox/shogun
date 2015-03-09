/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Parijat Mazumdar
 * Written (W) 2014 Saurabh Mahindre
 */

#include <shogun/clustering/KMeansMiniBatch.h>
#include <shogun/clustering/KMeansMiniBatchImpl.h>
#include <shogun/mathematics/Math.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

namespace shogun
{

CKMeansMiniBatch::CKMeansMiniBatch() : CKMeansBase()
{
	init();
}

CKMeansMiniBatch::CKMeansMiniBatch(int32_t k, CDistance* d, bool use_kmpp) : CKMeansBase(k, d, use_kmpp)
{
	init();
}

CKMeansMiniBatch::CKMeansMiniBatch(int32_t k_i, CDistance* d_i, SGMatrix<float64_t> centers_i) : CKMeansBase(k_i, d_i, centers_i)
{
	init();
}

CKMeansMiniBatch::~CKMeansMiniBatch()
{
}

void CKMeansMiniBatch::set_mbKMeans_batch_size(int32_t b)
{
	REQUIRE(b>0, "Parameter bach size should be > 0");
	m_batch_size=b;
}

int32_t CKMeansMiniBatch::get_mbKMeans_batch_size() const
{
	return m_batch_size;
}

void CKMeansMiniBatch::set_mbKMeans_iter(int32_t i)
{
	REQUIRE(i>0, "Parameter number of iterations should be > 0");
	m_minib_iter=i;
}

int32_t CKMeansMiniBatch::get_mbKMeans_iter() const
{
	return m_minib_iter;
}

void CKMeansMiniBatch::set_mbKMeans_params(int32_t b, int32_t t)
{
	REQUIRE(b>0, "Parameter bach size should be > 0");
	REQUIRE(t>0, "Parameter number of iterations should be > 0");
	m_batch_size=b;
	m_minib_iter=t;
}

bool CKMeansMiniBatch::train_machine(CFeatures* data)
{
	int32_t XSize;
	XSize=initialize_training(data);	
	
	SGVector<int32_t> cl_list=SGVector<int32_t>(XSize);
	cl_list.zero();
	SGVector<float64_t> weights_set=SGVector<float64_t>(m_k);
	weights_set.zero();

	if (m_mus_initial.matrix)
		set_initial_centers(weights_set, cl_list, XSize);
	else
		set_random_centers(weights_set, cl_list, XSize);

	CKMeansMiniBatchImpl::minibatch_KMeans(m_k, distance, m_batch_size, m_minib_iter, m_mus);

	compute_cluster_variances();
	return true;
}

void CKMeansMiniBatch::init()
{
	m_batch_size=-1;
	m_minib_iter=-1;
}

}
