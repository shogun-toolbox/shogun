/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Pan Deng, Wu Lin, Viktor Gal
 */

#include <shogun/kernel/ExponentialARDKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

CExponentialARDKernel::CExponentialARDKernel() : CDotKernel()
{
	init();
}

CExponentialARDKernel::~CExponentialARDKernel()
{
	CKernel::cleanup();
}

void CExponentialARDKernel::init()
{
	m_ARD_type=KT_SCALAR;

	m_log_weights=SGVector<float64_t>(1);
	m_log_weights.set_const(0.0);

	m_weights_rows=1.0;
	m_weights_cols=1.0;


	SG_ADD(&m_log_weights, "log_weights", "Feature weights in log domain", MS_AVAILABLE,
			GRADIENT_AVAILABLE);

	SG_ADD(&m_weights_rows, "weights_rows", "Row of feature weights", MS_NOT_AVAILABLE);
	SG_ADD(&m_weights_cols, "weights_cols", "Column of feature weights", MS_NOT_AVAILABLE);
	SG_ADD((int *)(&m_ARD_type), "type", "ARD kernel type", MS_NOT_AVAILABLE);

	m_weights_raw=SGMatrix<float64_t>();
	SG_ADD(&m_weights_raw, "weights_raw", "Features weights in standard domain", MS_NOT_AVAILABLE);

}

SGVector<float64_t> CExponentialARDKernel::get_feature_vector(int32_t idx, CFeatures* hs)
{
	REQUIRE(hs, "Features not set!\n");
	CDenseFeatures<float64_t> * dense_hs=dynamic_cast<CDenseFeatures<float64_t> *>(hs);
	if (dense_hs)
		return dense_hs->get_feature_vector(idx);

	CDotFeatures * dot_hs=dynamic_cast<CDotFeatures *>(hs);
	REQUIRE(dot_hs, "Kernel only supports DotFeatures\n");
	return dot_hs->get_computed_dot_feature_vector(idx);

}


void CExponentialARDKernel::set_weights(SGMatrix<float64_t> weights)
{
	REQUIRE(weights.num_rows>0 && weights.num_cols>0, "Weights matrix is non-empty\n");
	if (weights.num_rows==1)
	{
		if(weights.num_cols>1)
		{
			SGVector<float64_t> vec(weights.matrix,weights.num_cols,false);
			set_vector_weights(vec);
		}
		else
			set_scalar_weights(weights[0]);
	}
	else
		set_matrix_weights(weights);
}

void CExponentialARDKernel::lazy_update_weights()
{
	if (parameter_hash_changed())
	{
		if (m_ARD_type==KT_SCALAR || m_ARD_type==KT_DIAG)
		{
			SGMatrix<float64_t> log_weights(m_log_weights.vector,1,m_log_weights.vlen,false);
			m_weights_raw = linalg::exponent(m_log_weights);
		}
		else if (m_ARD_type==KT_FULL)
		{
			m_weights_raw=SGMatrix<float64_t>(m_weights_rows,m_weights_cols);
			m_weights_raw.set_const(0.0);
			index_t offset=0;
			for (int i=0;i<m_weights_raw.num_cols && i<m_weights_raw.num_rows;i++)
			{
				float64_t* begin=m_weights_raw.get_column_vector(i);
				std::copy(m_log_weights.vector+offset,m_log_weights.vector+offset+m_weights_raw.num_rows-i,begin+i);
				begin[i]=CMath::exp(begin[i]);
				offset+=m_weights_raw.num_rows-i;
			}
		}
		else
		{
			SG_ERROR("Unsupported ARD type\n");
		}
		update_parameter_hash();
	}
}

SGMatrix<float64_t> CExponentialARDKernel::get_weights()
{
	lazy_update_weights();
	return SGMatrix<float64_t>(m_weights_raw);
}

void CExponentialARDKernel::set_scalar_weights(float64_t weight)
{
	REQUIRE(weight>0, "Scalar (%f) weight should be positive\n",weight);
	m_log_weights=SGVector<float64_t>(1);
	m_log_weights.set_const(std::log(weight));
	m_ARD_type=KT_SCALAR;

	m_weights_rows=1.0;
	m_weights_cols=1.0;
}

void CExponentialARDKernel::set_vector_weights(SGVector<float64_t> weights)
{
	REQUIRE(rhs==NULL && lhs==NULL,
		"Setting vector weights must be before initialize features\n");
	REQUIRE(weights.vlen>0, "Vector weight should be non-empty\n");
	m_log_weights=SGVector<float64_t>(weights.vlen);
	for(index_t i=0; i<weights.vlen; i++)
	{
		REQUIRE(weights[i]>0, "Each entry of vector weight (v[%d]=%f) should be positive\n",
			i,weights[i]);
		m_log_weights[i] = std::log(weights[i]);
	}
	m_ARD_type=KT_DIAG;

	m_weights_rows=1.0;
	m_weights_cols=weights.vlen;
}

void CExponentialARDKernel::set_matrix_weights(SGMatrix<float64_t> weights)
{
	REQUIRE(rhs==NULL && lhs==NULL,
		"Setting matrix weights must be before initialize features\n");
	REQUIRE(weights.num_cols>0, "Matrix weight should be non-empty");
	REQUIRE(weights.num_rows>=weights.num_cols,
		"Number of row (%d) must be not less than number of column (%d)",
		weights.num_rows, weights.num_cols);

	m_weights_rows=weights.num_rows;
	m_weights_cols=weights.num_cols;
	m_ARD_type=KT_FULL;
	index_t len=(2*m_weights_rows+1-m_weights_cols)*m_weights_cols/2;
	m_log_weights=SGVector<float64_t>(len);

	index_t offset=0;
	for (int i=0; i<weights.num_cols && i<weights.num_rows; i++)
	{
		float64_t* begin=weights.get_column_vector(i);
		REQUIRE(begin[i]>0, "The diagonal entry of matrix weight (w(%d,%d)=%f) should be positive\n",
			i,i,begin[i]);
		std::copy(begin+i,begin+weights.num_rows,m_log_weights.vector+offset);
		m_log_weights[offset] = std::log(m_log_weights[offset]);
		offset+=weights.num_rows-i;
	}
}

CExponentialARDKernel::CExponentialARDKernel(int32_t size) : CDotKernel(size)
{
	init();
}

CExponentialARDKernel::CExponentialARDKernel(CDotFeatures* l,
		CDotFeatures* r, int32_t size)	: CDotKernel(size)
{
	init();
	init(l,r);
}

bool CExponentialARDKernel::init(CFeatures* l, CFeatures* r)
{
	cleanup();
	CDotKernel::init(l, r);
	int32_t dim=((CDotFeatures*) l)->get_dim_feature_space();
	if (m_ARD_type==KT_FULL)
	{
		REQUIRE(m_weights_rows==dim, "Dimension mismatch between features (%d) and weights (%d)\n",
			dim, m_weights_rows);
	}
	else if (m_ARD_type==KT_DIAG)
	{
		REQUIRE(m_log_weights.vlen==dim, "Dimension mismatch between features (%d) and weights (%d)\n",
			dim, m_log_weights.vlen);
	}
	return init_normalizer();
}


SGMatrix<float64_t> CExponentialARDKernel::get_weighted_vector(SGVector<float64_t> vec)
{
	REQUIRE(m_ARD_type==KT_FULL || m_ARD_type==KT_DIAG, "This method only supports vector weights or matrix weights\n");
	SGMatrix<float64_t> res;
	if (m_ARD_type==KT_FULL)
	{
		res=SGMatrix<float64_t>(m_weights_cols,1);
		index_t offset=0;
		// TODO: investigate a better way to make this
		// block thread-safe
		SGVector<float64_t> log_weights = m_log_weights.clone();
		//can be done it in parallel
		for (index_t i=0;i<m_weights_rows && i<m_weights_cols;i++)
		{
			SGMatrix<float64_t> weights(log_weights.vector+offset,1,m_weights_rows-i,false);
			weights[0]=CMath::exp(weights[0]);
			SGMatrix<float64_t> rtmp(vec.vector+i,vec.vlen-i,1,false);
			SGMatrix<float64_t> s=linalg::matrix_prod(weights,rtmp);
			weights[0] = std::log(weights[0]);
			res[i]=s[0];
			offset+=m_weights_rows-i;
		}
	}
	else
	{
		SGMatrix<float64_t> rtmp(vec.vector,vec.vlen,1,false);
		SGMatrix<float64_t> weights(linalg::exponent(m_log_weights));
		res = linalg::element_prod(weights, rtmp);
	}
	return res;
}

SGMatrix<float64_t> CExponentialARDKernel::compute_right_product(SGVector<float64_t>vec,
	float64_t & scalar_weight)
{
	SGMatrix<float64_t> right;

	if (m_ARD_type==KT_SCALAR)
	{
		right=SGMatrix<float64_t>(vec.vector,vec.vlen,1,false);
		scalar_weight*=CMath::exp(m_log_weights[0]);
	}
	else if (m_ARD_type==KT_DIAG || m_ARD_type==KT_FULL)
		right=get_weighted_vector(vec);
	else
	{
		SG_ERROR("Unsupported ARD type\n");
	}
	return right;
}

void CExponentialARDKernel::check_weight_gradient_index(index_t index)
{
	REQUIRE(lhs, "Left features not set!\n");
	REQUIRE(rhs, "Right features not set!\n");

	if (m_ARD_type!=KT_SCALAR)
	{
		REQUIRE(index>=0, "Index (%d) must be non-negative\n",index);
		REQUIRE(index<m_log_weights.vlen, "Index (%d) must be within #dimension of weights (%d)\n",
			index, m_log_weights.vlen);
	}
}
