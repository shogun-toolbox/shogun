/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Written (W) 2011 Abhinav Maurya
 * Written (W) 2012 Heiko Strathmann
 * Written (W) 2016 Soumyajit De
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CGaussianKernel::CGaussianKernel() : CShiftInvariantKernel()
{
	register_params();
}

CGaussianKernel::CGaussianKernel(float64_t w) : CShiftInvariantKernel()
{
	register_params();
	set_width(w);
}

CGaussianKernel::CGaussianKernel(int32_t size, float64_t w) : CShiftInvariantKernel()
{
	register_params();
	set_width(w);
	set_cache_size(size);
}

CGaussianKernel::CGaussianKernel(CDotFeatures* l, CDotFeatures* r, float64_t w, int32_t size) : CShiftInvariantKernel()
{
	register_params();
	set_width(w);
	set_cache_size(size);
	init(l, r);
}

CGaussianKernel::~CGaussianKernel()
{
	cleanup();
}

CGaussianKernel* CGaussianKernel::obtain_from_generic(CKernel* kernel)
{
	REQUIRE(kernel->get_kernel_type()==K_GAUSSIAN,
		"Provided kernel (%s) must be of type CGaussianKernel!\n", kernel->get_name());

	SG_REF(kernel);
	return (CGaussianKernel*)kernel;
}

#include <typeinfo>
CSGObject *CGaussianKernel::shallow_copy() const
{
	// TODO: remove this after all the classes get shallow_copy properly implemented
	// this assert is to avoid any subclass of CGaussianKernel accidentally called
	// with the implement here
	ASSERT(typeid(*this) == typeid(CGaussianKernel))
	CGaussianKernel *ker = new CGaussianKernel(cache_size, get_width());
	if (lhs && rhs)
	{
		ker->init(lhs, rhs);
		ker->m_distance->init(lhs, rhs);
	}
	return ker;
}

void CGaussianKernel::cleanup()
{
	CKernel::cleanup();
	m_distance->cleanup();
}

bool CGaussianKernel::init(CFeatures* l, CFeatures* r)
{
	cleanup();
	CShiftInvariantKernel::init(l, r);
	return init_normalizer();
}

void CGaussianKernel::set_width(float64_t w)
{
	REQUIRE(w>0, "width (%f) must be positive\n",w);
	m_log_width=CMath::log(w/2.0)/2.0;
}

float64_t CGaussianKernel::get_width() const
{
	return CMath::exp(m_log_width*2.0)*2.0;
}

SGMatrix<float64_t> CGaussianKernel::get_parameter_gradient(const TParameter* param, index_t index)
{
	REQUIRE(lhs, "Left hand side features must be set!\n")
	REQUIRE(rhs, "Rightt hand side features must be set!\n")

	if (!strcmp(param->m_name, "log_width"))
	{
		SGMatrix<float64_t> derivative=SGMatrix<float64_t>(num_lhs, num_rhs);
		for (int k=0; k<num_rhs; k++)
		{
#pragma omp parallel for
			for (int j=0; j<num_lhs; j++)
			{
				float64_t element=distance(j, k);
				derivative(j, k)=CMath::exp(-element)*element*2.0;
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

float64_t CGaussianKernel::compute(int32_t idx_a, int32_t idx_b)
{
    float64_t result=distance(idx_a, idx_b);
    return CMath::exp(-result);
}

void CGaussianKernel::load_serializable_post() throw (ShogunException)
{
	CKernel::load_serializable_post();
	if (lhs && rhs)
		m_distance->init(lhs, rhs);
}

float64_t CGaussianKernel::distance(int32_t idx_a, int32_t idx_b) const
{
	const float64_t inv_width=1.0/get_width();
	return CShiftInvariantKernel::distance(idx_a, idx_b)*inv_width;
}

void CGaussianKernel::register_params()
{
	set_width(1.0);
	set_cache_size(10);

	CEuclideanDistance* dist=new CEuclideanDistance();
	dist->set_disable_sqrt(true);
	m_distance=dist;
	SG_REF(m_distance);

	SG_ADD(&m_log_width, "log_width", "Kernel width in log domain", MS_AVAILABLE, GRADIENT_AVAILABLE);
}
