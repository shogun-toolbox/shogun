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

#include <shogun/kernel/LinearARDKernel.h>

#ifdef HAVE_LINALG_LIB
#include <shogun/mathematics/linalg/linalg.h>
#endif

using namespace shogun;

CLinearARDKernel::CLinearARDKernel() : CDotKernel()
{
	initialize();
}

CLinearARDKernel::~CLinearARDKernel()
{
	CKernel::cleanup();
}

void CLinearARDKernel::initialize()
{
	m_ARD_type=KT_SCALAR;
	m_weights=SGMatrix<float64_t>(1,1);
	m_weights.set_const(1.0);
	SG_ADD(&m_weights, "weights", "Feature weights", MS_AVAILABLE,
			GRADIENT_AVAILABLE);
	SG_ADD((int *)(&m_ARD_type), "type", "ARD kernel type", MS_NOT_AVAILABLE);
}

#ifdef HAVE_LINALG_LIB
CLinearARDKernel::CLinearARDKernel(int32_t size) : CDotKernel(size)
{
	initialize();
}

CLinearARDKernel::CLinearARDKernel(CDotFeatures* l,
		CDotFeatures* r, int32_t size)	: CDotKernel(size)
{
	initialize();
	init(l,r);
}

bool CLinearARDKernel::init(CFeatures* l, CFeatures* r)
{
	cleanup();
	CDotKernel::init(l, r);
	int32_t dim=((CDotFeatures*) l)->get_dim_feature_space();
	if (m_ARD_type==KT_FULL)
	{
		REQUIRE(m_weights.num_cols==dim, "Dimension mismatch between features (%d) and weights (%d)\n",
			dim, m_weights.num_cols);
	}
	else if (m_ARD_type==KT_DIAG)
	{
		REQUIRE(m_weights.num_rows==dim, "Dimension mismatch between features (%d) and weights (%d)\n",
			dim, m_weights.num_rows);
	}
	return init_normalizer();
}


SGMatrix<float64_t> CLinearARDKernel::compute_right_product(SGVector<float64_t>right_vec,
	float64_t & scalar_weight)
{
	SGMatrix<float64_t> right;

	if (m_ARD_type==KT_SCALAR)
	{
		right=SGMatrix<float64_t>(right_vec.vector,right_vec.vlen,1,false);
		scalar_weight*=m_weights[0];
	}
	else
	{
		SGMatrix<float64_t> rtmp(right_vec.vector,right_vec.vlen,1,false);

		if(m_ARD_type==KT_DIAG)
			right=linalg::elementwise_product(m_weights, rtmp);
		else if(m_ARD_type==KT_FULL)
			right=linalg::matrix_product(m_weights, rtmp);
		else
			SG_ERROR("Unsupported ARD type\n");
	}
	return right;
}

float64_t CLinearARDKernel::compute_helper(SGVector<float64_t> avec, SGVector<float64_t>bvec)
{
	SGMatrix<float64_t> left;
	SGMatrix<float64_t> left_transpose;
	float64_t scalar_weight=1.0;
	if (m_ARD_type==KT_SCALAR)
	{
		left=SGMatrix<float64_t>(avec.vector,1,avec.vlen,false);
		scalar_weight=m_weights[0];
	}
	else
	{
		SGMatrix<float64_t> ltmp(avec.vector,avec.vlen,1,false);
		if(m_ARD_type==KT_DIAG)
			left_transpose=linalg::elementwise_product(m_weights, ltmp);
		else if(m_ARD_type==KT_FULL)
			left_transpose=linalg::matrix_product(m_weights, ltmp);
		else
			SG_ERROR("Unsupported ARD type\n");
		left=SGMatrix<float64_t>(left_transpose.matrix,1,left_transpose.num_rows,false);
	}
	SGMatrix<float64_t> right=compute_right_product(bvec, scalar_weight);
	SGMatrix<float64_t> res=linalg::matrix_product(left, right);
	return res[0]*scalar_weight;
}

float64_t CLinearARDKernel::compute(int32_t idx_a, int32_t idx_b)
{
	REQUIRE(lhs, "Left features not set!\n");
	REQUIRE(rhs, "Right features not set!\n");
	SGVector<float64_t> avec=((CDotFeatures *)lhs)->get_computed_dot_feature_vector(idx_a);
	SGVector<float64_t> bvec=((CDotFeatures *)rhs)->get_computed_dot_feature_vector(idx_b);

	return compute_helper(avec, bvec);
}

float64_t CLinearARDKernel::compute_gradient_helper(SGVector<float64_t> avec,
	SGVector<float64_t> bvec, float64_t scale, index_t index)
{
	float64_t result;

	if(m_ARD_type==KT_DIAG)
	{
		result=2.0*avec[index]*bvec[index]*m_weights[index];
	}
	else
	{
		SGMatrix<float64_t> left(avec.vector,1,avec.vlen,false);
		SGMatrix<float64_t> right(bvec.vector,bvec.vlen,1,false);
		SGMatrix<float64_t> res;

		if (m_ARD_type==KT_SCALAR)
		{
			res=linalg::matrix_product(left, right);
			result=2.0*res[0]*m_weights[0];
		}
		else if(m_ARD_type==KT_FULL)
		{
			int32_t row_index=index%m_weights.num_rows;
			int32_t col_index=index/m_weights.num_rows;
			//index is a linearized index of m_weights (column-major)
			//m_weights is a d-by-p matrix, where p is #dimension of features
			SGVector<float64_t> row_vec=m_weights.get_row_vector(row_index);
			SGMatrix<float64_t> row_vec_r(row_vec.vector,row_vec.vlen,1,false);

			res=linalg::matrix_product(left, row_vec_r);
			result=res[0]*bvec[col_index];

			SGMatrix<float64_t> row_vec_l(row_vec.vector,1,row_vec.vlen,false);
			res=linalg::matrix_product(row_vec_l, right);
			result+=res[0]*avec[col_index];

		}
		else
		{
			SG_ERROR("Unsupported ARD type\n");
		}

	}
	return result*scale;
}


SGMatrix<float64_t> CLinearARDKernel::get_parameter_gradient(
	const TParameter* param, index_t index)
{
	REQUIRE(lhs, "Left features not set!\n");
	REQUIRE(rhs, "Right features not set!\n");

	int32_t row_index, col_index;
	if (m_ARD_type!=KT_SCALAR)
	{
		REQUIRE(index>=0, "Index (%d) must be non-negative\n",index);
		if (m_ARD_type==KT_DIAG)
		{
			REQUIRE(index<m_weights.num_rows, "Index (%d) must be within #dimension of weights (%d)\n",
				index, m_weights.num_rows);
		}
		else if(m_ARD_type==KT_FULL)
		{
			row_index=index%m_weights.num_rows;
			col_index=index/m_weights.num_rows;
			REQUIRE(row_index<m_weights.num_rows,
				"Row index (%d) must be within #row of weights (%d)\n",
				row_index, m_weights.num_rows);
			REQUIRE(col_index<m_weights.num_cols,
				"Column index (%d) must be within #column of weights (%d)\n",
				col_index, m_weights.num_cols);
		}
	}
	if (!strcmp(param->m_name, "weights"))
	{
		SGMatrix<float64_t> derivative(num_lhs, num_rhs);

		for (index_t j=0; j<num_lhs; j++)
		{
			SGVector<float64_t> avec=((CDotFeatures *)lhs)->get_computed_dot_feature_vector(j);
			for (index_t k=0; k<num_rhs; k++)
			{
				SGVector<float64_t> bvec=((CDotFeatures *)rhs)->get_computed_dot_feature_vector(k);
				derivative(j,k)=compute_gradient_helper(avec, bvec, 1.0, index);
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

SGMatrix<float64_t> CLinearARDKernel::get_weights()
{
	return SGMatrix<float64_t>(m_weights);
}

void CLinearARDKernel::set_weights(SGMatrix<float64_t> weights)
{
	REQUIRE(weights.num_cols>0 && weights.num_rows>0,
		"Weight Matrix (%d-by-%d) must not be empty\n",
		weights.num_rows, weights.num_cols);
	if (weights.num_cols>1)
	{
		m_ARD_type=KT_FULL;
	}
	else
	{
		if (weights.num_rows==1)
		{
			m_ARD_type=KT_SCALAR;
		}
		else
		{
			m_ARD_type=KT_DIAG;
		}
	}
	m_weights=weights;
}

void CLinearARDKernel::set_scalar_weights(float64_t weight)
{
	SGMatrix<float64_t> weights(1,1);
	weights(0,0)=weight;
	set_weights(weights);
}

void CLinearARDKernel::set_vector_weights(SGVector<float64_t> weights)
{
	SGMatrix<float64_t> weights_mat(weights.vlen,1);
	std::copy(weights.vector, weights.vector+weights.vlen, weights_mat.matrix);
	set_weights(weights_mat);
}

void CLinearARDKernel::set_matrix_weights(SGMatrix<float64_t> weights)
{
	set_weights(weights);
}
#endif //HAVE_LINALG_LIB
