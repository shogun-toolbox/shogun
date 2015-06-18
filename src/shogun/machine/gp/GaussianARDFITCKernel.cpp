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

#include <shogun/machine/gp/GaussianARDFITCKernel.h>

#ifdef HAVE_LINALG_LIB
#include <shogun/mathematics/linalg/linalg.h>
#endif

using namespace shogun;

CGaussianARDFITCKernel::CGaussianARDFITCKernel() : CGaussianARDKernel()
{
	initialize_kernel();
}
void CGaussianARDFITCKernel::initialize_kernel()
{
}

CGaussianARDFITCKernel::~CGaussianARDFITCKernel()
{
}

#if defined(HAVE_EIGEN3) && defined(HAVE_LINALG_LIB)
using namespace Eigen;

CGaussianARDFITCKernel::CGaussianARDFITCKernel(int32_t size, float64_t width)
		: CGaussianARDKernel(size,width)
{
	initialize_kernel();
}

CGaussianARDFITCKernel::CGaussianARDFITCKernel(CDotFeatures* l,
		CDotFeatures* r, int32_t size, float64_t width)
		: CGaussianARDKernel(l, r, size, width)
{
	initialize_kernel();
}

CGaussianARDFITCKernel* CGaussianARDFITCKernel::obtain_from_generic(CKernel* kernel)
{
	if (kernel->get_kernel_type()!=K_GAUSSIANARDFITC)
	{
		SG_SERROR("Provided kernel is not of type CGaussianARDFITCKernel!\n");
	}

	/* since an additional reference is returned */
	SG_REF(kernel);
	return (CGaussianARDFITCKernel*)kernel;
}

SGVector<float64_t> CGaussianARDFITCKernel::get_parameter_gradient_diagonal(
		const TParameter* param, index_t index)
{
	REQUIRE(param, "Param not set\n");
	if (!strcmp(param->m_name, "inducing_features"))
		return CKernel::get_parameter_gradient_diagonal(param, index);
	else
		return CGaussianARDKernel::get_parameter_gradient_diagonal(param, index);
}

SGMatrix<float64_t> CGaussianARDFITCKernel::get_parameter_gradient(
		const TParameter* param, index_t index)
{
	REQUIRE(param, "Param not set\n");
	if (!strcmp(param->m_name, "inducing_features"))
	{
		REQUIRE(lhs, "Left features not set!\n");
		REQUIRE(rhs, "Right features not set!\n");
		REQUIRE(index>=0 && index<num_lhs,"Index (%d) is out of bound (%d)\n",
			index, num_rhs);
		int32_t idx_l=index;
		//Note that CDotKernel requires lhs and rhs are CDotFeatures pointers
		//This Kernel is a subclass of CDotKernel
		SGVector<float64_t> left_vec=get_feature_vector(idx_l, lhs);
		SGMatrix<float64_t> res(left_vec.vlen, num_rhs);

		for (int32_t idx_r=0; idx_r<num_rhs; idx_r++)
		{
			SGVector<float64_t> right_vec=get_feature_vector(idx_r, rhs);
			SGMatrix<float64_t> res_transpose(res.get_column_vector(idx_r),1,left_vec.vlen,false);
			Map<MatrixXd> eigen_res_transpose(res_transpose.matrix, res_transpose.num_rows, res_transpose.num_cols);

			SGVector<float64_t> vec=linalg::add(left_vec, right_vec, 1.0, -1.0);
			float64_t scalar_weight=1.0;
			//column vector
			SGMatrix<float64_t> right=compute_right_product(vec, scalar_weight);
			//row vector
			SGMatrix<float64_t> right_transpose(right.matrix,1,right.num_rows,false);
			Map<MatrixXd> eigen_right_transpose(right_transpose.matrix, right_transpose.num_rows, right_transpose.num_cols);
			if (m_ARD_type==KT_SCALAR)
			{
				scalar_weight*=m_weights[0];
				eigen_res_transpose=eigen_right_transpose*scalar_weight;
			}
			else
			{
				if(m_ARD_type==KT_DIAG)
				{
					Map<MatrixXd> eigen_weights(m_weights.matrix, 1, m_weights.num_rows);
					eigen_res_transpose=eigen_right_transpose.cwiseProduct(eigen_weights);
				}
				else if(m_ARD_type==KT_FULL)
				{
					Map<MatrixXd> eigen_weights(m_weights.matrix, m_weights.num_rows, m_weights.num_cols);
					eigen_res_transpose=eigen_right_transpose*eigen_weights;
				}
				else
				{
					SG_ERROR("Unsupported ARD type\n");
				}

			}
			for (index_t i=0; i<left_vec.vlen; i++)
				res(i,idx_r)*=kernel(idx_l,idx_r)*-2.0/m_width;
		}
		return res;
	}
	else
	{
		return CGaussianARDKernel::get_parameter_gradient(param, index);
	}
}
#endif /* HAVE_LINALG_LIB */

