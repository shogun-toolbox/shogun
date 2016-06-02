/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Heiko Strathmann
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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
 */

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>

#include "KernelExpFamilyNystromHImpl.h"

using namespace shogun;
using namespace Eigen;

KernelExpFamilyNystromHImpl::KernelExpFamilyNystromHImpl(SGMatrix<float64_t> data, float64_t sigma, float64_t lambda,
		SGVector<index_t> inds, bool low_memory_mode)  : KernelExpFamilyNystromImpl(data, sigma, lambda, inds, low_memory_mode)
{
}

KernelExpFamilyNystromHImpl::KernelExpFamilyNystromHImpl(SGMatrix<float64_t> data, float64_t sigma, float64_t lambda,
		index_t num_rkhs_basis, bool low_memory_mode)  : KernelExpFamilyNystromImpl(data, sigma, lambda, num_rkhs_basis, low_memory_mode)
{
}

std::pair<SGMatrix<float64_t>, SGVector<float64_t>> KernelExpFamilyNystromHImpl::build_system() const
{
	auto D = get_num_dimensions();
	auto N = get_num_data_lhs();
	auto ND = N*D;
	auto m = get_num_rkhs_basis();

	SG_SINFO("Allocating memory for system.\n");
	SGMatrix<float64_t> A(m+1, m+1);
	Map<MatrixXd> eigen_A(A.matrix, m+1, m+1);
	SGVector<float64_t> b(m+1);
	Map<VectorXd> eigen_b(b.vector, m+1);

	SG_SINFO("Computing h.\n");
	auto h = compute_h();
	auto eigen_h=Map<VectorXd>(h.vector, ND);

	SG_SINFO("Computing xi norm.\n");
	auto xi_norm_2 = compute_xi_norm_2();

	SG_SINFO("Computing all kernel Hessians.\n");
	auto all_hessians = kernel_hessian_all();
	auto eigen_hessians = Map<MatrixXd>(all_hessians.matrix, ND, ND);

	SG_SINFO("Creating sub-sampled copies.\n");
	SGMatrix<float64_t> col_sub_sampled_hessian(ND, m);
	SGMatrix<float64_t> sub_sampled_hessian(m, m);
	SGVector<float64_t> sub_sampled_h(m);

	auto eigen_col_sub_sampled_hessian = Map<MatrixXd>(col_sub_sampled_hessian.matrix, ND, m);
	auto eigen_sub_sampled_hessian = Map<MatrixXd>(sub_sampled_hessian.matrix, m, m);
	auto eigen_sub_sampled_h = Map<VectorXd>(sub_sampled_h.vector, m);

	for (auto i=0; i<m; i++)
	{
		memcpy(col_sub_sampled_hessian.get_column_vector(i),
				all_hessians.get_column_vector(m_inds[i]),
				sizeof(float64_t)*ND);

		for (auto j=0; j<m; j++)
			sub_sampled_hessian(i,j)=eigen_hessians(m_inds[i], m_inds[j]);

		sub_sampled_h[i] = h[m_inds[i]];
	}

	SG_SINFO("Populating A matrix.\n");
	A(0,0) = eigen_h.squaredNorm() / N + m_lambda * xi_norm_2;

	// can use noalias to speed up as matrices are definitely different
	eigen_A.block(1,1,m,m).noalias()=eigen_col_sub_sampled_hessian.transpose()*eigen_col_sub_sampled_hessian / N + m_lambda*eigen_sub_sampled_hessian;
	eigen_A.col(0).segment(1, m).noalias() = eigen_sub_sampled_hessian*eigen_sub_sampled_h / N + m_lambda*eigen_sub_sampled_h;

	for (auto ind_idx=0; ind_idx<m; ind_idx++)
		A(0, ind_idx+1) = A(ind_idx+1, 0);

	b[0] = -xi_norm_2;
	eigen_b.segment(1, m) = -eigen_sub_sampled_h;

	return std::pair<SGMatrix<float64_t>, SGVector<float64_t>>(A, b);
}
