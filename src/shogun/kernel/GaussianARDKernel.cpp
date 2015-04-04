/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Wu Lin
 * Written (W) 2012 Jacob Walker
 *
 * Adapted from WeightedDegreeRBFKernel.cpp
 */

#include <shogun/kernel/GaussianARDKernel.h>
#include <shogun/mathematics/Math.h>

#ifdef HAVE_LINALG_LIB
#include <shogun/mathematics/linalg/linalg.h>
#endif

using namespace shogun;

CGaussianARDKernel::CGaussianARDKernel() : CLinearARDKernel()
{
	initialize();
}

CGaussianARDKernel::~CGaussianARDKernel()
{
}

void CGaussianARDKernel::initialize()
{
	set_width(1.0);
	SG_ADD(&m_width, "width", "Kernel width", MS_AVAILABLE, GRADIENT_AVAILABLE);
}

#ifdef HAVE_LINALG_LIB
CGaussianARDKernel::CGaussianARDKernel(int32_t size, float64_t width)
		: CLinearARDKernel(size)
{
	initialize();
	set_width(width);
}

CGaussianARDKernel::CGaussianARDKernel(CDotFeatures* l,
		CDotFeatures* r, int32_t size, float64_t width)
		: CLinearARDKernel(size)
{
	initialize();
	set_width(width);
}

bool CGaussianARDKernel::init(CFeatures* l, CFeatures* r)
{
	return CLinearARDKernel::init(l,r);
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
	float64_t result=distance(idx_a,idx_b);
	return CMath::exp(-result);
}

SGVector<float64_t> CGaussianARDKernel::get_parameter_gradient_diagonal(
		const TParameter* param, index_t index)
{
	REQUIRE(param, "Param not set\n");
	REQUIRE(lhs , "Left features not set!\n");
	REQUIRE(rhs, "Right features not set!\n");

	if (lhs==rhs) 
	{
		if (!strcmp(param->m_name, "weights") || !strcmp(param->m_name, "width"))
		{
			SGVector<float64_t> derivative(num_lhs);
			derivative.zero();
			return derivative;
		}
	}
	else
	{
		int32_t length=CMath::min(num_lhs, num_rhs);
		SGVector<float64_t> derivative(length);

		for (index_t j=0; j<length; j++)
		{
			if (!strcmp(param->m_name, "weights") )
			{
				SGVector<float64_t> avec=((CDotFeatures *)lhs)->get_computed_dot_feature_vector(j);
				SGVector<float64_t> bvec=((CDotFeatures *)rhs)->get_computed_dot_feature_vector(j);
				derivative[j]=get_parameter_gradient_helper(param,index,j,j,avec,bvec);
			}
			else if (!strcmp(param->m_name, "width"))
			{
				SGVector<float64_t> avec, bvec;
				derivative[j]=get_parameter_gradient_helper(param,index,j,j,avec,bvec);
			}
		}
		return derivative;
	}

	SG_ERROR("Can't compute derivative wrt %s parameter\n", param->m_name);
	return SGVector<float64_t>();
}


float64_t CGaussianARDKernel::get_parameter_gradient_helper(
	const TParameter* param, index_t index, int32_t idx_a,
	int32_t idx_b, SGVector<float64_t> avec, SGVector<float64_t> bvec)
{
	REQUIRE(param, "Param not set\n");

	if (!strcmp(param->m_name, "weights"))
	{
		linalg::add(avec, bvec, bvec, 1.0, -1.0);
		float64_t scale=-kernel(idx_a,idx_b)/m_width;
		return	compute_gradient_helper(bvec, bvec, scale, index);
	}
	else if (!strcmp(param->m_name, "width"))
	{
		float64_t tmp=kernel(idx_a,idx_b);
		return -tmp*CMath::log(tmp)/m_width;
	}
	else
	{
		SG_ERROR("Can't compute derivative wrt %s parameter\n", param->m_name);
		return 0.0;
	}
}

SGMatrix<float64_t> CGaussianARDKernel::get_parameter_gradient(
		const TParameter* param, index_t index)
{
	REQUIRE(param, "Param not set\n");
	REQUIRE(lhs , "Left features not set!\n");
	REQUIRE(rhs, "Right features not set!\n");

	if (!strcmp(param->m_name, "weights"))
	{
		SGMatrix<float64_t> derivative(num_lhs, num_rhs);
		for (index_t j=0; j<num_lhs; j++)
		{
			SGVector<float64_t> avec=((CDotFeatures *)lhs)->get_computed_dot_feature_vector(j);
			for (index_t k=0; k<num_rhs; k++)
			{
				SGVector<float64_t> bvec=((CDotFeatures *)rhs)->get_computed_dot_feature_vector(k);
				derivative(j,k)=get_parameter_gradient_helper(param,index,j,k,avec,bvec);
			}
		}
		return derivative;
	}
	else if (!strcmp(param->m_name, "width"))
	{
		SGMatrix<float64_t> derivative(num_lhs, num_rhs);

		for (index_t j=0; j<num_lhs; j++)
		{
			for (index_t k=0; k<num_rhs; k++)
			{
				SGVector<float64_t> avec, bvec;
				derivative(j,k)=get_parameter_gradient_helper(param,index,j,k,avec,bvec);
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

float64_t CGaussianARDKernel::distance(int32_t idx_a, int32_t idx_b)
{
	REQUIRE(lhs, "Left features (lhs) not set!\n")
	REQUIRE(rhs, "Right features (rhs) not set!\n")

	if (lhs==rhs && idx_a==idx_b) 
		return 0.0;

	SGVector<float64_t> avec=((CDotFeatures *)lhs)->get_computed_dot_feature_vector(idx_a);
	SGVector<float64_t> bvec=((CDotFeatures *)rhs)->get_computed_dot_feature_vector(idx_b);
	linalg::add(avec, bvec, avec, 1.0, -1.0);
	float64_t result=compute_helper(avec, avec);
	return result/m_width;
}
#endif /* HAVE_LINALG_LIB */
