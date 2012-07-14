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
#include <shogun/kernel/GaussianARDKernel.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CGaussianARDKernel::CGaussianARDKernel()
: CDotKernel()
{
	init();
}


CGaussianARDKernel::CGaussianARDKernel(int32_t size, float64_t width)
: CDotKernel(size), m_width(width)
{
	init();
}

CGaussianARDKernel::CGaussianARDKernel(CDenseFeatures<float64_t>* l,
		CDenseFeatures<float64_t>* r,
		int32_t size, float64_t width)
: CDotKernel(size), m_width(width)
{
	init();
	init(l,r);
}

void CGaussianARDKernel::init()
{
	m_width = 2.0;
	m_weights = SGVector<float64_t>();

	SG_ADD(&m_weights, "weights", "Feature Weights", MS_AVAILABLE);
	SG_ADD(&m_width, "width", "Kernel Width", MS_AVAILABLE);
}

CGaussianARDKernel::~CGaussianARDKernel()
{
	CKernel::cleanup();
}

bool CGaussianARDKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);

	init_ft_weights();

	SG_DEBUG("Initialized GaussianARDKernel (%p).\n", this);

	return init_normalizer();
}

void CGaussianARDKernel::init_ft_weights()
{
	int32_t alen, blen;

	alen = ((CDenseFeatures<float64_t>*) lhs)->get_num_features();
	blen = ((CDenseFeatures<float64_t>*) rhs)->get_num_features();

	REQUIRE(alen==blen, "Number of Right and Left Hand "\
			"Features Must be the Same./n");

	m_weights = SGVector<float64_t>(alen);

	for (int32_t i=0; i < alen; i++)
		m_weights[i]=1.0;

	SG_DEBUG("Initialized weights for LinearARDKernel (%p).\n", this);
}

void CGaussianARDKernel::set_weight(float64_t w, index_t i)
{
	if (i > m_weights.vlen-1)
	{
		SG_ERROR("Index %i out of range for LinearARDKernel."\
				 "Number of features is %i.\n", i, m_weights.vlen);
	}

	m_weights[i]=w;
}

float64_t CGaussianARDKernel::get_weight(index_t i)
{
	if (i > m_weights.vlen-1)
	{
		SG_ERROR("Index %i out of range for LinearARDKernel."\
				 "Number of features is %i.\n", i, m_weights.vlen);
	}

	return m_weights[i];
}

float64_t CGaussianARDKernel::compute(int32_t idx_a, int32_t idx_b)
{
	SGVector<float64_t> avec
		= ((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a);
	SGVector<float64_t> bvec
		= ((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b);

	REQUIRE(avec.vlen==bvec.vlen, "Number of Right and Left Hand "\
			"Features Must be the Same./n");

	float64_t result=0;

	for (index_t i = 0; i < avec.vlen; i++)
		result += CMath::pow((avec[i]-bvec[i])*m_weights[i], 2);

	return CMath::exp(-result/m_width);
}
