/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 */

#include <shogun/machine/gp/GaussianARDSparseKernel.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <utility>

using namespace shogun;

GaussianARDSparseKernel::GaussianARDSparseKernel() : GaussianARDKernel()
{
	initialize_sparse_kernel();
}
void GaussianARDSparseKernel::initialize_sparse_kernel()
{
}

GaussianARDSparseKernel::~GaussianARDSparseKernel()
{
}

using namespace Eigen;

GaussianARDSparseKernel::GaussianARDSparseKernel(int32_t size)
		: GaussianARDKernel(size)
{
	initialize_sparse_kernel();
}

GaussianARDSparseKernel::GaussianARDSparseKernel(std::shared_ptr<DotFeatures> l,
		std::shared_ptr<DotFeatures> r, int32_t size)
		: GaussianARDKernel(std::move(l), std::move(r), size)
{
	initialize_sparse_kernel();
}

std::shared_ptr<GaussianARDSparseKernel> GaussianARDSparseKernel::obtain_from_generic(const std::shared_ptr<Kernel>& kernel)
{
	if (kernel->get_kernel_type()!=K_GAUSSIANARDSPARSE)
	{
		error("Provided kernel is not of type GaussianARDSparseKernel!");
	}

	/* since an additional reference is returned */

	return kernel->as<GaussianARDSparseKernel>();
}

SGVector<float64_t> GaussianARDSparseKernel::get_parameter_gradient_diagonal(
		Parameters::const_reference param, index_t index)
{
	if (param.first == "inducing_features")
		return Kernel::get_parameter_gradient_diagonal(param, index);
	else
		return GaussianARDKernel::get_parameter_gradient_diagonal(param, index);
}

SGMatrix<float64_t> GaussianARDSparseKernel::get_parameter_gradient(
		Parameters::const_reference param, index_t index)
{
	if (param.first == "inducing_features")
	{
		require(lhs, "Left features not set!");
		require(rhs, "Right features not set!");
		require(index>=0 && index<num_lhs,"Index ({}) is out of bound ({})",
			index, num_rhs);
		int32_t idx_l=index;
		//Note that DotKernel requires lhs and rhs are DotFeatures pointers
		//This Kernel is a subclass of DotKernel
		SGVector<float64_t> left_vec=get_feature_vector(idx_l, lhs);
		SGMatrix<float64_t> res(left_vec.vlen, num_rhs);

		lazy_update_weights();

		for (int32_t idx_r=0; idx_r<num_rhs; idx_r++)
		{
			SGVector<float64_t> right_vec=get_feature_vector(idx_r, rhs);
			Map<VectorXd> eigen_res_col_vec(res.get_column_vector(idx_r),left_vec.vlen);

			SGVector<float64_t> vec=linalg::add(left_vec, right_vec, 1.0, -1.0);
			float64_t scalar_weight=1.0;
			//column vector
			SGMatrix<float64_t> right=compute_right_product(vec, scalar_weight);
			Map<VectorXd> eigen_right_col_vec(right.matrix,right.num_rows);

			if (m_ARD_type==KT_SCALAR)
			{
				scalar_weight*=m_weights_raw[0];
				eigen_res_col_vec=eigen_right_col_vec*scalar_weight;
			}
			else
			{
				if(m_ARD_type==KT_DIAG)
				{
					Map<VectorXd> eigen_weights(m_weights_raw.matrix, m_log_weights.vlen);
					eigen_res_col_vec=eigen_right_col_vec.cwiseProduct(eigen_weights);
				}
				else if(m_ARD_type==KT_FULL)
				{
					Map<MatrixXd> eigen_weights(m_weights_raw.matrix, m_weights_raw.num_rows, m_weights_raw.num_cols);
					eigen_res_col_vec=eigen_weights*eigen_right_col_vec;
				}
				else
				{
					error("Unsupported ARD type");
				}

			}
			for (index_t i=0; i<left_vec.vlen; i++)
				res(i,idx_r)*=-kernel(idx_l,idx_r);
		}
		return res;
	}
	else
	{
		return GaussianARDKernel::get_parameter_gradient(param, index);
	}
}
