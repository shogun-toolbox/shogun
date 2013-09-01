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

#include <shogun/lib/common.h>
#include <shogun/kernel/GaussianARDKernel.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CGaussianARDKernel::CGaussianARDKernel() : CLinearARDKernel()
{
	init();
}


CGaussianARDKernel::CGaussianARDKernel(int32_t size, float64_t width)
		: CLinearARDKernel(size), m_width(width)
{
	init();
}

CGaussianARDKernel::CGaussianARDKernel(CDenseFeatures<float64_t>* l,
		CDenseFeatures<float64_t>* r, int32_t size, float64_t width)
		: CLinearARDKernel(size), m_width(width)
{
	init();
}

bool CGaussianARDKernel::init(CFeatures* l, CFeatures* r)
{
	return CLinearARDKernel::init(l,r);
}

void CGaussianARDKernel::init()
{
	m_width = 2.0;

	SG_ADD(&m_width, "width", "Kernel Width", MS_AVAILABLE);
}

CGaussianARDKernel::~CGaussianARDKernel()
{
}

CGaussianARDKernel* CGaussianARDKernel::obtain_from_generic(CKernel* kernel)
{
	if (kernel->get_kernel_type()!=K_GAUSSIANARD)
	{
		SG_SERROR("Provided kernel is not of type CGaussianARDKernel!\n");
	}

	/* since an additional reference is returned */
	SG_REF(kernel);
	return (CGaussianARDKernel*)kernel;
}

float64_t CGaussianARDKernel::compute(int32_t idx_a, int32_t idx_b)
{
	if (!lhs || !rhs)
		SG_ERROR("Features not set!\n")

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

SGMatrix<float64_t> CGaussianARDKernel::get_parameter_gradient(TParameter* param,
		index_t index)
{
	if (!lhs || !rhs)
		SG_ERROR("Features not set!\n")

	if (!strcmp(param->m_name, "weights"))
	{
		SGMatrix<float64_t> derivative = get_kernel_matrix();

		for (index_t j = 0; j < num_lhs; j++)
		{
			for (index_t k = 0; k < num_rhs; k++)
			{
				SGVector<float64_t> avec
					= ((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(j);
				SGVector<float64_t> bvec
					= ((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(k);

				REQUIRE(avec.vlen==bvec.vlen, "Number of Right and Left Hand "\
						"Features Must be the Same./n");

				float64_t element = compute(j,k);
				float64_t product =
						CMath::pow((avec[index]-bvec[index]), 2)
						*(m_weights[index]/m_width);

				derivative(j,k) = -2*element*product;
			}
		}

		return derivative;
	}
	else if (!strcmp(param->m_name, "width"))
	{
		SGMatrix<float64_t> derivative(num_lhs, num_rhs);

		for (index_t j = 0; j < num_lhs; j++)
		{
			for (index_t k = 0; k < num_rhs; k++)
			{
				SGVector<float64_t> avec
					= ((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(j);
				SGVector<float64_t> bvec
					= ((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(k);

				REQUIRE(avec.vlen==bvec.vlen, "Number of Right and Left Hand "\
						"Features Must be the Same./n");

				float64_t result=0;

				for (index_t i = 0; i < avec.vlen; i++)
					result += CMath::pow((avec[i]-bvec[i])*m_weights[i], 2);

				derivative(j,k) = CMath::exp(-result/m_width)*
						result/(m_width*m_width);
			}
		}

		return derivative;
	}
	else
		return SGMatrix<float64_t>();
}
