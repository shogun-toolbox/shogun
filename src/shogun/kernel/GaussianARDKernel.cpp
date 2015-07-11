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

CGaussianARDKernel::CGaussianARDKernel() : CExponentialARDKernel()
{
	init();
}

CGaussianARDKernel::~CGaussianARDKernel()
{
}

void CGaussianARDKernel::init()
{

#ifdef HAVE_LINALG_LIB
	m_sq_lhs=SGVector<float64_t>();
	m_sq_rhs=SGVector<float64_t>();
	SG_ADD(&m_sq_lhs, "sq_lhs", "squared left-hand side", MS_NOT_AVAILABLE);
	SG_ADD(&m_sq_rhs, "sq_rhs", "squared right-hand side", MS_NOT_AVAILABLE);
#endif
}

float64_t CGaussianARDKernel::distance(int32_t idx_a, int32_t idx_b)
{
	float64_t result=0.0;
#ifdef HAVE_LINALG_LIB
	REQUIRE(lhs, "Left features (lhs) not set!\n")
	REQUIRE(rhs, "Right features (rhs) not set!\n")

	if (lhs==rhs && idx_a==idx_b)
		return result;

	if (m_ARD_type==KT_SCALAR)
	{
		result=(m_sq_lhs[idx_a]+m_sq_rhs[idx_b]-2.0*CDotKernel::compute(idx_a,idx_b));
		result*=CMath::exp(2.0*m_log_weights[0]);
	}
	else
	{
		SGVector<float64_t> avec=get_feature_vector(idx_a, lhs);
		SGVector<float64_t> bvec=get_feature_vector(idx_b, rhs);
		avec=linalg::add(avec, bvec, 1.0, -1.0);
		result=compute_helper(avec, avec);
	}
#endif /* HAVE_LINALG_LIB */
	return result/2.0;
}

#ifdef HAVE_LINALG_LIB
CGaussianARDKernel::CGaussianARDKernel(int32_t size)
		: CExponentialARDKernel(size)
{
	init();
}

CGaussianARDKernel::CGaussianARDKernel(CDotFeatures* l,
		CDotFeatures* r, int32_t size)
		: CExponentialARDKernel(size)
{
	init();
}

bool CGaussianARDKernel::init(CFeatures* l, CFeatures* r)
{
	bool status=CExponentialARDKernel::init(l,r);

	if (m_ARD_type==KT_SCALAR)
		precompute_squared();

	return status;
}

SGVector<float64_t> CGaussianARDKernel::precompute_squared_helper(CDotFeatures* df)
{
	REQUIRE(df, "Features not set\n")
	int32_t num_vec=df->get_num_vectors();
	SGVector<float64_t> sq(num_vec);
	for (int32_t i=0; i<num_vec; i++)
		sq[i]=df->dot(i,df, i);
	return sq;
}

void CGaussianARDKernel::precompute_squared()
{
	if (!lhs || !rhs)
		return;
	SG_SPRINT("called\n");
	m_sq_lhs=precompute_squared_helper((CDotFeatures*) lhs);

	if (lhs==rhs)
		m_sq_rhs=m_sq_lhs;
	else
		m_sq_rhs=precompute_squared_helper((CDotFeatures*) rhs);
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

float64_t CGaussianARDKernel::compute_helper(SGVector<float64_t> avec, SGVector<float64_t>bvec)
{
	SGMatrix<float64_t> left;
	SGMatrix<float64_t> left_transpose;
	float64_t scalar_weight=1.0;
	if (m_ARD_type==KT_SCALAR)
	{
		left=SGMatrix<float64_t>(avec.vector,1,avec.vlen,false);
		scalar_weight=CMath::exp(m_log_weights[0]);
	}
	else if(m_ARD_type==KT_FULL || m_ARD_type==KT_DIAG)
	{
		left_transpose=get_weighted_vector(avec);
		left=SGMatrix<float64_t>(left_transpose.matrix,1,left_transpose.num_rows,false);
	}
	else
		SG_ERROR("Unsupported ARD type\n");
	SGMatrix<float64_t> right=compute_right_product(bvec, scalar_weight);
	SGMatrix<float64_t> res=linalg::matrix_product(left, right);
	return res[0]*scalar_weight;
}

float64_t CGaussianARDKernel::compute_gradient_helper(SGVector<float64_t> avec,
	SGVector<float64_t> bvec, float64_t scale, index_t index)
{
	float64_t result=0.0;

	if(m_ARD_type==KT_DIAG)
		result=2.0*avec[index]*bvec[index]*CMath::exp(2.0*m_log_weights[index]);
	else
	{
		SGMatrix<float64_t> res;

		if (m_ARD_type==KT_SCALAR)
		{
			SGMatrix<float64_t> left(avec.vector,1,avec.vlen,false);
			SGMatrix<float64_t> right(bvec.vector,bvec.vlen,1,false);
			res=linalg::matrix_product(left, right);
			result=2.0*res[0]*CMath::exp(2.0*m_log_weights[0]);
		}
		else if(m_ARD_type==KT_FULL)
		{
			int32_t row_index=0;
			int32_t col_index=index;
			int32_t offset=m_weights_rows;
			int32_t total_offset=0;
			while(col_index>=offset && offset>0)
			{
				col_index-=offset;
				total_offset+=offset;
				offset--;
				row_index++;
			}
			col_index+=row_index;

			SGVector<float64_t> row_vec=SGVector<float64_t>(m_log_weights.vector+total_offset,m_weights_rows-row_index,false);   
			row_vec[0]=CMath::exp(row_vec[0]);

			SGMatrix<float64_t> row_vec_r(row_vec.vector,row_vec.vlen,1,false);
			SGMatrix<float64_t> left(avec.vector+row_index,1,avec.vlen-row_index,false);

			res=linalg::matrix_product(left, row_vec_r);
			result=res[0]*bvec[col_index];

			SGMatrix<float64_t> row_vec_l(row_vec.vector,1,row_vec.vlen,false);
			SGMatrix<float64_t> right(bvec.vector+row_index,bvec.vlen-row_index,1,false);

			res=linalg::matrix_product(row_vec_l, right);
			result+=res[0]*avec[col_index];

			if(row_index==col_index)
				result*=row_vec[0];
			row_vec[0]=CMath::log(row_vec[0]);
		}
		else
		{
			SG_ERROR("Unsupported ARD type\n");
		}

	}
	return result*scale;
}


SGVector<float64_t> CGaussianARDKernel::get_parameter_gradient_diagonal(
		const TParameter* param, index_t index)
{
	REQUIRE(param, "Param not set\n");
	REQUIRE(lhs , "Left features not set!\n");
	REQUIRE(rhs, "Right features not set!\n");

	if (lhs==rhs)
	{
		if (!strcmp(param->m_name, "log_weights"))
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
		check_weight_gradient_index(index);
		for (index_t j=0; j<length; j++)
		{
			if (!strcmp(param->m_name, "log_weights") )
			{
				if (m_ARD_type==KT_SCALAR)
				{
					float64_t dist=distance(j,j);
					derivative[j]=CMath::exp(-dist)*(-dist*2.0);
				}
				else
				{
					SGVector<float64_t> avec=get_feature_vector(j, lhs);
					SGVector<float64_t> bvec=get_feature_vector(j, rhs);
					derivative[j]=get_parameter_gradient_helper(param,index,j,j,avec,bvec);
				}

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

	if (!strcmp(param->m_name, "log_weights"))
	{
		bvec=linalg::add(avec, bvec, 1.0, -1.0);
		float64_t scale=-kernel(idx_a,idx_b)/2.0;
		return	compute_gradient_helper(bvec, bvec, scale, index);
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

	if (!strcmp(param->m_name, "log_weights"))
	{
		SGMatrix<float64_t> derivative(num_lhs, num_rhs);
		check_weight_gradient_index(index);
		for (index_t j=0; j<num_lhs; j++)
		{
			SGVector<float64_t> avec=get_feature_vector(j, lhs);
			for (index_t k=0; k<num_rhs; k++)
			{
				if (m_ARD_type==KT_SCALAR)
				{
					float64_t dist=distance(j,k);
					derivative(j,k)=CMath::exp(-dist)*(-dist*2.0);
				}
				else
				{
					SGVector<float64_t> bvec=get_feature_vector(k, rhs);
					derivative(j,k)=get_parameter_gradient_helper(param,index,j,k,avec,bvec);
				}
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
#endif /* HAVE_LINALG_LIB */
