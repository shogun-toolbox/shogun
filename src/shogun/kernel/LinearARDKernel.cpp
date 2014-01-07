/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Jacob Walker
 *
 * Adapted from WeightedDegreeRBFKernel.cpp
 */

#include <lib/common.h>
#include <kernel/LinearARDKernel.h>
#include <features/Features.h>
#include <io/SGIO.h>

using namespace shogun;

CLinearARDKernel::CLinearARDKernel() : CDotKernel()
{
	init();
}


CLinearARDKernel::CLinearARDKernel(int32_t size) : CDotKernel(size)
{
	init();
}

CLinearARDKernel::CLinearARDKernel(CDenseFeatures<float64_t>* l,
		CDenseFeatures<float64_t>* r, int32_t size)	: CDotKernel(size)
{
	init();
	init(l,r);
}

void CLinearARDKernel::init()
{
	SG_ADD(&m_weights, "weights", "Feature weights", MS_AVAILABLE,
			GRADIENT_AVAILABLE);
}

CLinearARDKernel::~CLinearARDKernel()
{
	CKernel::cleanup();
}

bool CLinearARDKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);

	init_ft_weights();

	return init_normalizer();
}

void CLinearARDKernel::init_ft_weights()
{
	if (!lhs || !rhs)
		return;

	int32_t alen, blen;

	alen=((CDenseFeatures<float64_t>*) lhs)->get_num_features();
	blen=((CDenseFeatures<float64_t>*) rhs)->get_num_features();

	REQUIRE(alen==blen, "Number of right and left hand features must be the "
			"same\n")

	if (m_weights.vlen != alen)
	{
		m_weights=SGVector<float64_t>(alen);
		m_weights.set_const(1.0);
	}

	SG_DEBUG("Initialized weights for LinearARDKernel (%p).\n", this)
}

void CLinearARDKernel::set_weight(float64_t w, index_t i)
{
	if (i >= m_weights.vlen)
	{
		SG_ERROR("Index %i out of range for LinearARDKernel."\
				 "Number of features is %i.\n", i, m_weights.vlen);
	}

	m_weights[i]=w;
}

float64_t CLinearARDKernel::get_weight(index_t i)
{
	if (i >= m_weights.vlen)
	{
		SG_ERROR("Index %i out of range for LinearARDKernel."\
				 "Number of features is %i.\n", i, m_weights.vlen);
	}

	return m_weights[i];
}

float64_t CLinearARDKernel::compute(int32_t idx_a, int32_t idx_b)
{
	REQUIRE(lhs && rhs, "Features not set!\n")

	SGVector<float64_t> avec=
		((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a);
	SGVector<float64_t> bvec=
		((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b);

	REQUIRE(avec.vlen==bvec.vlen, "Number of right and left hand "
			"features must be the same\n")

	float64_t result=0;

	for (index_t i = 0; i < avec.vlen; i++)
		result += avec[i]*bvec[i]*m_weights[i]*m_weights[i];

	return result;
}

SGMatrix<float64_t> CLinearARDKernel::get_parameter_gradient(
		const TParameter* param, index_t index)
{
	REQUIRE(lhs && rhs, "Features not set!\n")

	if (!strcmp(param->m_name, "weights"))
	{
		SGMatrix<float64_t> derivative(num_lhs, num_rhs);

		for (index_t j=0; j<num_lhs; j++)
		{
			for (index_t k=0; k<num_rhs; k++)
			{
				SGVector<float64_t> avec=
					((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(j);
				SGVector<float64_t> bvec=
					((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(k);

				REQUIRE(avec.vlen==bvec.vlen, "Number of right and left hand "
						"features must be the same\n");

				derivative(j,k)=avec[index]*bvec[index]*m_weights[index];
			}
		}
		return derivative;
	}
	else
	{
		SG_ERROR("Can't compute derivative wrt %s parameter\n", param->m_name);
		return SGMatrix<float64_t>();
	}
}
