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

#include "KernelExpFamilyNystromImpl.h"

using namespace shogun;
using namespace Eigen;

KernelExpFamilyNystromImpl::KernelExpFamilyNystromImpl(SGMatrix<float64_t> data, float64_t sigma, float64_t lambda,
		SGVector<index_t> rkhs_basis_inds)  : KernelExpFamilyImpl(data, sigma, lambda)
{
	m_rkhs_basis_inds = rkhs_basis_inds;

	SG_SINFO("Using m=%d user-defined RKHS basis functions.\n", rkhs_basis_inds.vlen);
}


KernelExpFamilyNystromImpl::KernelExpFamilyNystromImpl(SGMatrix<float64_t> data, float64_t sigma, float64_t lambda,
		index_t num_rkhs_basis)  : KernelExpFamilyImpl(data, sigma, lambda)
{
	sub_sample_rkhs_basis(num_rkhs_basis);
}

void KernelExpFamilyNystromImpl::sub_sample_rkhs_basis(index_t num_rkhs_basis)
{
	SG_SINFO("Using m=%d uniformly sampled RKHS basis function.\n", num_rkhs_basis);
	auto N = get_num_data_lhs();
	auto D = get_num_dimensions();

	SGVector<index_t> permutation(N*D);
	permutation.range_fill();
	CMath::permute(permutation);
	m_rkhs_basis_inds = SGVector<index_t>(num_rkhs_basis);
	for (auto i=0; i<num_rkhs_basis; i++)
		m_rkhs_basis_inds[i]=permutation[i];

	// in order to have more sequential data reads
	CMath::qsort(m_rkhs_basis_inds.vector, num_rkhs_basis);
}

index_t KernelExpFamilyNystromImpl::get_num_rkhs_basis() const
{
	return m_rkhs_basis_inds.vlen;
}

std::pair<index_t, index_t> KernelExpFamilyNystromImpl::idx_to_ai(index_t idx) const
{
	auto D = get_num_dimensions();
	return std::pair<index_t, index_t>(idx / D, idx % D);
}

float64_t KernelExpFamilyNystromImpl::difference_component(index_t idx_a, index_t idx_b, index_t i) const
{
	auto D = get_num_dimensions();

	if (m_differences.matrix)
	{
		auto N_rhs = m_data_rhs.matrix ? get_num_data_rhs(): get_num_data_lhs();
		auto diff = SGVector<float64_t>(m_differences.get_column_vector(idx_a*N_rhs+idx_b), D, false);
		return diff[i];
	}

	Map<VectorXd> x(m_data_lhs.get_column_vector(idx_a), D);
	float64_t* right_pointer = m_data_rhs.matrix ?
			m_data_rhs.get_column_vector(idx_b) : m_data_lhs.get_column_vector(idx_b);
	Map<VectorXd> y(right_pointer, D);

	return y[i]-x[i];
}

float64_t KernelExpFamilyNystromImpl::kernel_hessian_component(const index_t idx_a, const index_t idx_b, const index_t i, const index_t j) const
{
	auto D = get_num_dimensions();

	Map<VectorXd> x(m_data_lhs.get_column_vector(idx_a), D);
	Map<VectorXd> y(m_data_lhs.get_column_vector(idx_b), D);

	//k = gaussian_kernel(x_2d, y_2d, sigma)
	auto k=kernel(idx_a, idx_b);

	auto differences_i = y[i] - x[i];
	auto differences_j = y[j] - x[j];

	float64_t ridge = 0;
	if (i==j)
	{
		ridge = 2;
		ridge /= m_sigma;
	}

	return k*(ridge - 4*(differences_i*differences_j)/pow(m_sigma, 2));
}

float64_t KernelExpFamilyNystromImpl::kernel_dx_dx_dy_dy_component(index_t idx_a, index_t idx_b, index_t i, index_t j) const
{
	// this assumes that distances are precomputed, i.e. this call only causes memory io
	auto diff2_i = pow(difference_component(idx_a, idx_b, i), 2);
	auto diff2_j = pow(difference_component(idx_a, idx_b, j), 2);

	auto k=kernel(idx_a,idx_b);
	auto factor = k*pow(2.0/m_sigma, 3);

	float64_t result = k*pow(2.0/m_sigma, 4) * (diff2_i*diff2_j);
	result -= factor*(diff2_i+diff2_j - 1);
	if (i==j)
		result -= 4*factor*diff2_i - 2*factor;

	return result;
}

float64_t KernelExpFamilyNystromImpl::compute_xi_norm_2() const
{
	auto N = get_num_data_lhs();
	auto m = get_num_rkhs_basis();
	auto D = get_num_dimensions();
	float64_t xi_norm_2=0;

#pragma omp parallel for reduction (+:xi_norm_2)
	for (auto idx_a=0; idx_a<N; idx_a++)
		for (auto i=0; i<D; i++)
			for (auto col_idx=0; col_idx<m; col_idx++)
			{
				auto bj = idx_to_ai(m_rkhs_basis_inds[col_idx]);
				auto idx_b = bj.first;
				auto j = bj.second;
				xi_norm_2 += kernel_dx_dx_dy_dy_component(idx_a, idx_b, i, j);
			}

	// TODO check math as the number of terms is different here
	xi_norm_2 /= (N*N);

	return xi_norm_2;
}

std::pair<SGMatrix<float64_t>, SGVector<float64_t>> KernelExpFamilyNystromImpl::build_system() const
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

	// TODO dont compute full h
	SG_SINFO("Computing h.\n");
	auto h = compute_h();
	auto eigen_h=Map<VectorXd>(h.vector, m);

	SG_SINFO("Computing xi norm.\n");
	auto xi_norm_2 = compute_xi_norm_2();

	SG_SINFO("Creating sub-sampled kernel Hessians.\n");
	SGMatrix<float64_t> col_sub_sampled_hessian(ND, m);
	SGMatrix<float64_t> sub_sampled_hessian(m, m);

	auto eigen_col_sub_sampled_hessian = Map<MatrixXd>(col_sub_sampled_hessian.matrix, ND, m);
	auto eigen_sub_sampled_hessian = Map<MatrixXd>(sub_sampled_hessian.matrix, m, m);

#pragma omp parallel for
	for (auto col_idx=0; col_idx<m; col_idx++)
	{
		auto bj = idx_to_ai(m_rkhs_basis_inds[col_idx]);
		auto idx_b = bj.first;
		auto j = bj.second;

		// TODO compute the whole column of all kernel hessians at once
		for (auto row_idx=0; row_idx<ND; row_idx++)
		{
			auto ai = idx_to_ai(row_idx);
			auto idx_a = ai.first;
			auto i = ai.second;
			col_sub_sampled_hessian(row_idx, col_idx)=
					kernel_hessian_component(idx_a, idx_b, i, j);
		}

		for (auto row_idx=0; row_idx<m; row_idx++)
			sub_sampled_hessian(row_idx,col_idx)=col_sub_sampled_hessian(m_rkhs_basis_inds[row_idx], col_idx);
	}

	SG_SINFO("Populating A matrix.\n");
	A(0,0) = eigen_h.squaredNorm() / N + m_lambda * xi_norm_2;

	// can use noalias to speed up as matrices are definitely different
	eigen_A.block(1,1,m,m).noalias()=eigen_col_sub_sampled_hessian.transpose()*eigen_col_sub_sampled_hessian / N + m_lambda*eigen_sub_sampled_hessian;
	eigen_A.col(0).segment(1, m).noalias() = eigen_sub_sampled_hessian*eigen_h / N + m_lambda*eigen_h;

	for (auto ind_idx=0; ind_idx<m; ind_idx++)
		A(0, ind_idx+1) = A(ind_idx+1, 0);

	b[0] = -xi_norm_2;
	eigen_b.segment(1, m) = -eigen_h;

	return std::pair<SGMatrix<float64_t>, SGVector<float64_t>>(A, b);
}

SGVector<float64_t> KernelExpFamilyNystromImpl::compute_h() const
{
	auto m = get_num_rkhs_basis();
	auto D = get_num_dimensions();
	auto N = get_num_data_lhs();

	SGVector<float64_t> h(m);
	Map<VectorXd> eigen_h(h.vector, m);
	eigen_h = VectorXd::Zero(m);

#pragma omp parallel for
	for (auto rkhs_idx=0; rkhs_idx<m; rkhs_idx++)
	{
		auto bj = idx_to_ai(m_rkhs_basis_inds[rkhs_idx]);
		auto idx_b = bj.first;
		auto j = bj.second;

		// TODO compute sum in single go
		for (auto idx_a=0; idx_a<N; idx_a++)
			for (auto i=0; i<D; i++)
				h[rkhs_idx] += kernel_dx_dx_dy_component(idx_a, idx_b, i, j);
	}

	eigen_h /= N;

	return h;
}

float64_t KernelExpFamilyNystromImpl::kernel_dx_dx_dy_component(index_t idx_a, index_t idx_b, index_t i, index_t j) const
{
	// this assumes that distances are precomputed, i.e. this call only causes memory io
	SGVector<float64_t> diff=difference(idx_a, idx_b);
	auto diff2_i = pow(diff[i], 2);

	auto k=kernel(idx_a,idx_b);

	float64_t result = -pow(2./m_sigma,3) * k * diff2_i*diff[j];

	if (i==j)
		result += pow(2./m_sigma,2) * k * 2* diff[i];

	result += pow(2./m_sigma,2) * k * diff[j];

	return result;
}

float64_t KernelExpFamilyNystromImpl::kernel_dx_component(index_t idx_a, index_t idx_b, index_t i) const
{
	//k = gaussian_kernel(x_2d, y_2d, sigma)
	auto diff_i = difference_component(idx_a, idx_b, i);
	auto k=kernel(idx_a, idx_b);

	return 2*k*diff_i/m_sigma;
}

float64_t KernelExpFamilyNystromImpl::kernel_dx_dx_component(index_t idx_a, index_t idx_b, index_t i) const
{
	auto diff_i = difference_component(idx_a, idx_b, i);
	auto k=kernel(idx_a, idx_b);

	return k*(pow(diff_i,2)*pow(2.0/m_sigma, 2) -2.0/m_sigma);
}

SGVector<float64_t> KernelExpFamilyNystromImpl::kernel_dx_i_dx_i_dx_j_component(index_t idx_a, index_t idx_b, index_t i) const
{
	auto D = get_num_dimensions();
	auto diff = difference(idx_a, idx_b);
	auto eigen_diff = Map<VectorXd>(diff.vector, D);
	auto k=kernel(idx_a, idx_b);

	SGVector<float64_t> result(D);
	Map<VectorXd> eigen_result(result.vector, D);

	eigen_result = pow(eigen_diff[i],2)*eigen_diff.transpose();
	eigen_result *= k* pow(2.0/m_sigma, 3);
	eigen_result -= k * eigen_diff.transpose() * pow(2.0/m_sigma, 2);
	eigen_result[i] -= 2* k * eigen_diff[i] * pow(2.0/m_sigma, 2);

	return result;
}

SGVector<float64_t> KernelExpFamilyNystromImpl::kernel_dx_i_dx_j_component(index_t idx_a, index_t idx_b, index_t i) const
{
	auto D = get_num_dimensions();
	auto diff = difference(idx_a, idx_b);
	auto eigen_diff = Map<VectorXd>(diff.vector, D);
	auto k=kernel(idx_a, idx_b);

	SGVector<float64_t> result(D);
	Map<VectorXd> eigen_result(result.vector, D);

	eigen_result = eigen_diff[i]*eigen_diff;
	eigen_result *= k * pow(2.0/m_sigma, 2);
	eigen_result[i] -= k * 2.0/m_sigma;

	return result;
}

float64_t KernelExpFamilyNystromImpl::log_pdf(index_t idx_test) const
{
	auto N = get_num_data_lhs();
	auto m = get_num_rkhs_basis();

	float64_t xi = 0;
	float64_t beta_sum = 0;

	for (auto ind_idx=0; ind_idx<m; ind_idx++)
	{
		auto ai = idx_to_ai(m_rkhs_basis_inds[ind_idx]);
		auto a = ai.first;
		auto i = ai.second;

		auto xi_grad_i = kernel_dx_dx_component(a, idx_test, i);
		auto grad_x_xa_i = kernel_dx_component(a, idx_test, i);

		xi += xi_grad_i;
		// note: sign flip due to swapped kernel arugment compared to Python code
		beta_sum -= grad_x_xa_i * m_alpha_beta[1+ind_idx];
	}

	return m_alpha_beta[0]*xi/N + beta_sum;
}

SGVector<float64_t> KernelExpFamilyNystromImpl::grad(index_t idx_test) const
{
	auto N = get_num_data_lhs();
	auto D = get_num_dimensions();
	auto m = get_num_rkhs_basis();

	SGVector<float64_t> xi_grad(D);
	SGVector<float64_t> beta_sum_grad(D);
	Map<VectorXd> eigen_xi_grad(xi_grad.vector, D);
	Map<VectorXd> eigen_beta_sum_grad(beta_sum_grad.vector, D);
	eigen_xi_grad = VectorXd::Zero(D);
	eigen_beta_sum_grad.array() = VectorXd::Zero(D);

	for (auto ind_idx=0; ind_idx<m; ind_idx++)
	{
		auto ai = idx_to_ai(m_rkhs_basis_inds[ind_idx]);
		auto a = ai.first;
		auto i = ai.second;

		auto xi_gradient_mat_component = kernel_dx_i_dx_i_dx_j_component(a, idx_test, i);
		Map<VectorXd> eigen_xi_gradient_mat_component(xi_gradient_mat_component.vector, D);
		auto left_arg_hessian_component = kernel_dx_i_dx_j_component(a, idx_test, i);
		Map<VectorXd> eigen_left_arg_hessian_component(left_arg_hessian_component.vector, D);

		// note: sign flip due to swapped kernel argument compared to Python code
		eigen_xi_grad -= eigen_xi_gradient_mat_component;
		eigen_beta_sum_grad += eigen_left_arg_hessian_component * m_alpha_beta[1+ind_idx];
	}

	// re-use memory
	eigen_xi_grad *= m_alpha_beta[0] / N;
	eigen_xi_grad += eigen_beta_sum_grad;
	return xi_grad;
}

SGMatrix<float64_t> KernelExpFamilyNystromImpl::pinv_self_adjoint(const SGMatrix<float64_t>& A)
{
	// based on the snippet from
	// http://eigen.tuxfamily.org/index.php?title=FAQ#Is_there_a_method_to_compute_the_.28Moore-Penrose.29_pseudo_inverse_.3F
	// modified using eigensolver for psd problems
	auto m=A.num_rows;
	ASSERT(A.num_cols == m);
	auto eigen_A=Map<MatrixXd>(A.matrix, m, m);

	SelfAdjointEigenSolver<MatrixXd> solver(eigen_A);
	auto s = solver.eigenvalues();
	auto V = solver.eigenvectors();

	// tol = eps⋅max(m,n) * max(singularvalues)
	// this is done in numpy/Octave & co
	// c.f. https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse#Singular_value_decomposition_.28SVD.29
	float64_t pinv_tol = CMath::MACHINE_EPSILON * m * s.maxCoeff();

	VectorXd inv_s(m);
	for (auto i=0; i<m; i++)
	{
		if (s(i) > pinv_tol)
			inv_s(i)=1.0/s(i);
		else
			inv_s(i)=0;
	}

	SGMatrix<float64_t> A_pinv(m, m);
	Map<MatrixXd> eigen_pinv(A_pinv.matrix, m, m);
	eigen_pinv = (V*inv_s.asDiagonal()*V.transpose());

	return A_pinv;
}
