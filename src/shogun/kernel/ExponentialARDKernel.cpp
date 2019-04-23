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

ExponentialARDKernel::ExponentialARDKernel() : DotKernel()
{
	init();
}

ExponentialARDKernel::~ExponentialARDKernel()
{
	Kernel::cleanup();
}

void ExponentialARDKernel::init()
{
	m_ARD_type=KT_SCALAR;

	m_log_weights=SGVector<float64_t>(1);
	m_log_weights.set_const(0.0);

	m_weights_rows=1.0;
	m_weights_cols=1.0;


	SG_ADD(&m_log_weights, "log_weights", "Feature weights in log domain", ParameterProperties::HYPER |
	ParameterProperties::GRADIENT);

	SG_ADD(&m_weights_rows, "weights_rows", "Row of feature weights");
	SG_ADD(&m_weights_cols, "weights_cols", "Column of feature weights");
	SG_ADD((int *)(&m_ARD_type), "type", "ARD kernel type");

	m_weights_raw=SGMatrix<float64_t>();
	SG_ADD(&m_weights_raw, "weights_raw", "Features weights in standard domain");

}

SGVector<float64_t> ExponentialARDKernel::get_feature_vector(int32_t idx, std::shared_ptr<Features> hs)
{
	require(hs, "Features not set!");
	auto dense_hs=std::dynamic_pointer_cast<DenseFeatures<float64_t>>(hs);
	if (dense_hs)
		return dense_hs->get_feature_vector(idx);

	auto dot_hs=std::dynamic_pointer_cast<DotFeatures>(hs);
	require(dot_hs, "Kernel only supports DotFeatures");
	return dot_hs->get_computed_dot_feature_vector(idx);

}


void ExponentialARDKernel::set_weights(SGMatrix<float64_t> weights)
{
	require(weights.num_rows>0 && weights.num_cols>0, "Weights matrix is non-empty");
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

void ExponentialARDKernel::lazy_update_weights()
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
				begin[i] = std::exp(begin[i]);
				offset+=m_weights_raw.num_rows-i;
			}
		}
		else
		{
			error("Unsupported ARD type");
		}
		update_parameter_hash();
	}
}

SGMatrix<float64_t> ExponentialARDKernel::get_weights()
{
	lazy_update_weights();
	return SGMatrix<float64_t>(m_weights_raw);
}

void ExponentialARDKernel::set_scalar_weights(float64_t weight)
{
	require(weight>0, "Scalar ({}) weight should be positive",weight);
	m_log_weights=SGVector<float64_t>(1);
	m_log_weights.set_const(std::log(weight));
	m_ARD_type=KT_SCALAR;

	m_weights_rows=1.0;
	m_weights_cols=1.0;
}

void ExponentialARDKernel::set_vector_weights(SGVector<float64_t> weights)
{
	require(rhs==NULL && lhs==NULL,
		"Setting vector weights must be before initialize features");
	require(weights.vlen>0, "Vector weight should be non-empty");
	m_log_weights=SGVector<float64_t>(weights.vlen);
	for(index_t i=0; i<weights.vlen; i++)
	{
		require(weights[i]>0, "Each entry of vector weight (v[{}]={}) should be positive",
			i,weights[i]);
		m_log_weights[i] = std::log(weights[i]);
	}
	m_ARD_type=KT_DIAG;

	m_weights_rows=1.0;
	m_weights_cols=weights.vlen;
}

void ExponentialARDKernel::set_matrix_weights(SGMatrix<float64_t> weights)
{
	require(rhs==NULL && lhs==NULL,
		"Setting matrix weights must be before initialize features");
	require(weights.num_cols>0, "Matrix weight should be non-empty");
	require(weights.num_rows>=weights.num_cols,
		"Number of row ({}) must be not less than number of column ({})",
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
		require(begin[i]>0, "The diagonal entry of matrix weight (w({},{})={}) should be positive",
			i,i,begin[i]);
		std::copy(begin+i,begin+weights.num_rows,m_log_weights.vector+offset);
		m_log_weights[offset] = std::log(m_log_weights[offset]);
		offset+=weights.num_rows-i;
	}
}

ExponentialARDKernel::ExponentialARDKernel(int32_t size) : DotKernel(size)
{
	init();
}

ExponentialARDKernel::ExponentialARDKernel(std::shared_ptr<DotFeatures> l,
		std::shared_ptr<DotFeatures> r, int32_t size)	: DotKernel(size)
{
	init();
	init(l,r);
}

bool ExponentialARDKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	cleanup();
	DotKernel::init(l, r);
	int32_t dim=(std::static_pointer_cast<DotFeatures>(l))->get_dim_feature_space();
	if (m_ARD_type==KT_FULL)
	{
		require(m_weights_rows==dim, "Dimension mismatch between features ({}) and weights ({})",
			dim, m_weights_rows);
	}
	else if (m_ARD_type==KT_DIAG)
	{
		require(m_log_weights.vlen==dim, "Dimension mismatch between features ({}) and weights ({})",
			dim, m_log_weights.vlen);
	}
	return init_normalizer();
}


SGMatrix<float64_t> ExponentialARDKernel::get_weighted_vector(SGVector<float64_t> vec)
{
	require(m_ARD_type==KT_FULL || m_ARD_type==KT_DIAG, "This method only supports vector weights or matrix weights");
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
			weights[0] = std::exp(weights[0]);
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

SGMatrix<float64_t> ExponentialARDKernel::compute_right_product(SGVector<float64_t>vec,
	float64_t & scalar_weight)
{
	SGMatrix<float64_t> right;

	if (m_ARD_type==KT_SCALAR)
	{
		right=SGMatrix<float64_t>(vec.vector,vec.vlen,1,false);
		scalar_weight *= std::exp(m_log_weights[0]);
	}
	else if (m_ARD_type==KT_DIAG || m_ARD_type==KT_FULL)
		right=get_weighted_vector(vec);
	else
	{
		error("Unsupported ARD type");
	}
	return right;
}

void ExponentialARDKernel::check_weight_gradient_index(index_t index)
{
	require(lhs, "Left features not set!");
	require(rhs, "Right features not set!");

	if (m_ARD_type!=KT_SCALAR)
	{
		require(index>=0, "Index ({}) must be non-negative",index);
		require(index<m_log_weights.vlen, "Index ({}) must be within #dimension of weights ({})",
			index, m_log_weights.vlen);
	}
}
