/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Written (W) 2011 Abhinav Maurya
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <shogun/lib/common.h>
#include <shogun/base/Parameter.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/io/SGIO.h>

#ifdef USE_OPENCL
#include <shogun/opencl/kernels/svm/dot_kernels.h>
#endif 
using namespace shogun;

CGaussianKernel::CGaussianKernel()
	: CDotKernel()
{
	init();
}

CGaussianKernel::CGaussianKernel(int32_t size, float64_t w)
: CDotKernel(size)
{
	init();
	set_width(w);
}

CGaussianKernel::CGaussianKernel(
	CDotFeatures* l, CDotFeatures* r, float64_t w, int32_t size)
: CDotKernel(size)
{
	init();
	set_width(w);
	init(l,r);
}

CGaussianKernel::~CGaussianKernel()
{
	cleanup();
}

void CGaussianKernel::cleanup()
{
	if (sq_lhs != sq_rhs)
		SG_FREE(sq_rhs);
	sq_rhs = NULL;

	SG_FREE(sq_lhs);
	sq_lhs = NULL;

	CKernel::cleanup();
}

void CGaussianKernel::precompute_squared_helper(float64_t* &buf, CDotFeatures* df)
{
	ASSERT(df);
	int32_t num_vec=df->get_num_vectors();
	buf=SG_MALLOC(float64_t, num_vec);

	for (int32_t i=0; i<num_vec; i++)
		buf[i]=df->dot(i,df, i);
}

bool CGaussianKernel::init(CFeatures* l, CFeatures* r)
{
	///free sq_{r,l}hs first
	cleanup();

	CDotKernel::init(l, r);
	precompute_squared();
	return init_normalizer();
}

float64_t CGaussianKernel::compute(int32_t idx_a, int32_t idx_b)
{
	if (!m_compact)
	{
		float64_t result=sq_lhs[idx_a]+sq_rhs[idx_b]
				-2*CDotKernel::compute(idx_a, idx_b);
		return CMath::exp(-result/width);
	}

	int32_t len_features, power;
	len_features=((CSimpleFeatures<float64_t>*) lhs)->get_num_features();
	power=(len_features%2==0) ? (len_features+1):len_features;

	float64_t result=sq_lhs[idx_a]+sq_rhs[idx_b]-2*CDotKernel::compute(idx_a,idx_b);
	float64_t result_multiplier=1-(sqrt(result/width))/3;

	if (result_multiplier<=0)
		result_multiplier=0;
	else
		result_multiplier=pow(result_multiplier, power);

	return result_multiplier*exp(-result/width);
}

void CGaussianKernel::load_serializable_post() throw (ShogunException)
{
	CKernel::load_serializable_post();
	precompute_squared();
}

void CGaussianKernel::precompute_squared()
{
	if (!lhs || !rhs)
		return;

	precompute_squared_helper(sq_lhs, (CDotFeatures*) lhs);

	if (lhs==rhs)
		sq_rhs=sq_lhs;
	else
		precompute_squared_helper(sq_rhs, (CDotFeatures*) rhs);
}


#ifdef USE_OPENCL
void CGaussianKernel::ocl_compute(SGVector<int32_t> const & svs){
	dynamic_cast<CDotFeatures*>(lhs)->enqueue_ocl_dot_program(ocl_kernel_matrix,svs,dynamic_cast<CDotFeatures*>(rhs));
	viennacl::ocl::get_queue().finish();
	viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(shogun::ocl::svm::dot_kernels::program_name(lhs->get_feature_type()), "gaussian_kernel");
	viennacl::ocl::enqueue(k(cl_double(width),
				 viennacl::traits::handle(ocl_kernel_matrix), 
					cl_uint(viennacl::traits::size1(ocl_kernel_matrix)),          cl_uint(viennacl::traits::size2(ocl_kernel_matrix)),
					cl_uint(viennacl::traits::internal_size1(ocl_kernel_matrix)), cl_uint(viennacl::traits::internal_size2(ocl_kernel_matrix))));
	viennacl::ocl::get_queue().finish();
}
#endif
		
void CGaussianKernel::init()
{
	set_width(1.0);
	set_compact_enabled(false);
	sq_lhs=NULL;
	sq_rhs=NULL;
	SG_ADD(&width, "width", "Kernel width.", MS_AVAILABLE);
	SG_ADD(&m_compact, "compact", "Compact Enabled Option.", MS_AVAILABLE);
}
