/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * (W) 2012 Jacob Walker
 *
 * Adapted from WeightedDegreeRBFKernel.cpp
 *
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/LinearARDKernel.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CLinearARDKernel::CLinearARDKernel()
: CDotKernel(), m_dimension(1), m_weights(0)
{
	init();
}


CLinearARDKernel::CLinearARDKernel(int32_t size)
: CDotKernel(size), m_dimension(0), m_weights(0)
{
	init();
}

CLinearARDKernel::CLinearARDKernel(CDenseFeatures<float64_t>* l,
		CDenseFeatures<float64_t>* r,
		int32_t size)
: CDotKernel(size), m_weights(0)
{
	init();
	init(l,r);
}

void CLinearARDKernel::init()
{
	m_weights = NULL;
	m_dimension = 0;
	m_parameters->add_vector(&m_weights, &m_dimension, "weights");
}

CLinearARDKernel::~CLinearARDKernel()
{
	SG_FREE(m_weights);
	m_weights=NULL;
	CKernel::cleanup();
}

bool CLinearARDKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);

	int32_t alen, blen;

	alen = ((CDenseFeatures<float64_t>*) lhs)->get_num_features();
	blen = ((CDenseFeatures<float64_t>*) rhs)->get_num_features();

	ASSERT(alen==blen);

	m_dimension = alen;

	SG_DEBUG("Initialized LinearARDKernel (%p).\n", this);

	return (init_normalizer() && init_ft_weights());
}

bool CLinearARDKernel::init_ft_weights()
{
	ASSERT(m_dimension>0);

	if (m_weights!=0)
		SG_FREE(m_weights);

	m_weights=SG_MALLOC(float64_t, m_dimension);

	if (m_weights)
	{
		for (index_t i=0; i<m_dimension; i++)
			m_weights[i]=1.0;

		SG_DEBUG("Initialized weights for LinearARDKernel (%p).\n", this);
		return true;
	}

	else
		return false;
}

void CLinearARDKernel::set_weight(float64_t w, index_t i)
{
	if (i > m_dimension-1)
		SG_ERROR("Index %i out of range for LinearARDKernel."\
				 "Number of features is %i.\n", i, m_dimension);
	m_weights[i]=w;
}

float64_t CLinearARDKernel::get_weight(index_t i)
{
	if (i > m_dimension-1)
		SG_ERROR("Index %i out of range for LinearARDKernel."\
				 "Number of features is %i.\n", i, m_dimension);
	return m_weights[i];
}

float64_t CLinearARDKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=((CDenseFeatures<float64_t>*) lhs)->
			get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=((CDenseFeatures<float64_t>*) rhs)->
			get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen);

	float64_t result=0;

	for (index_t i = 0; i < m_dimension; i++)
		result += avec[i]*bvec[i]*m_weights[i]*m_weights[i];

	return result;
}
