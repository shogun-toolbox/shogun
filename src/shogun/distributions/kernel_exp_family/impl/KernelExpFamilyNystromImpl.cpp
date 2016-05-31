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
		SGVector<index_t> inds, bool low_memory_mode)  : KernelExpFamilyImpl(data, sigma, lambda)
{
	m_inds = inds;
	m_low_memory_mode = low_memory_mode;

	SG_SINFO("Using m=%d user-defined RKHS basis function.\n", inds.vlen);
}

KernelExpFamilyNystromImpl::KernelExpFamilyNystromImpl(SGMatrix<float64_t> data, float64_t sigma, float64_t lambda,
		index_t num_rkhs_basis, bool low_memory_mode)  : KernelExpFamilyImpl(data, sigma, lambda)
{
	sub_sample_rkhs_basis(num_rkhs_basis);
	m_low_memory_mode = low_memory_mode;
}

void KernelExpFamilyNystromImpl::sub_sample_rkhs_basis(index_t num_rkhs_basis)
{
	SG_SINFO("Using m=%d uniformly sampled RKHS basis function.\n", num_rkhs_basis);
	auto N = get_num_data_lhs();
	auto D = get_num_dimensions();

	SGVector<index_t> permutation(N*D);
	permutation.range_fill();
	CMath::permute(permutation);
	m_inds = SGVector<index_t>(num_rkhs_basis);
	for (auto i=0; i<num_rkhs_basis; i++)
		m_inds[i]=permutation[i];

	// in order to have more sequential data reads
	CMath::qsort(m_inds.vector, num_rkhs_basis);
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

std::pair<index_t, index_t> KernelExpFamilyNystromImpl::idx_to_ai(index_t idx) const
{
	auto D = get_num_dimensions();
	return std::pair<index_t, index_t>(idx / D, idx % D);
}

float64_t KernelExpFamilyNystromImpl::compute_lower_right_submatrix_element(index_t row_idx,
		index_t col_idx) const
{
	// TODO benchmark against full version using all kernel hessians
	auto D = get_num_dimensions();
	auto N = get_num_data_lhs();

	auto ai = idx_to_ai(row_idx);
	auto bj = idx_to_ai(col_idx);
	auto a = ai.first;
	auto i = ai.second;
	auto b = bj.first;
	auto j = bj.second;

	auto G_a_b_i_j = kernel_hessian_component(a, b, i, j);

	float64_t G_sum = 0;
	// TODO check parallel with accumulating on the G_sum
	// no parallel here as is called from build_system's parallel
	// TODO merge with compute_first_row_no_storing
	for (auto idx_n=0; idx_n<N; idx_n++)
		for (auto idx_d=0; idx_d<D; idx_d++)
		{
			// TODO find case when G1=G2 and dont re-compute
			auto G1 = kernel_hessian_component(a, idx_n, i, idx_d);
			auto G2 = kernel_hessian_component(idx_n, b, idx_d, j);
			G_sum += G1*G2;
		}

	return G_sum/N + m_lambda*G_a_b_i_j;
}

SGVector<float64_t> KernelExpFamilyNystromImpl::compute_first_row_no_storing() const
{
	// TODO benchmark against the first row in build_system
	auto D = get_num_dimensions();
	auto N = get_num_data_lhs();
	auto ND = N*D;

	auto h=compute_h();
	Map<VectorXd> eigen_h(h.vector, h.vlen);
	SGVector<float64_t> result(ND);
	Map<VectorXd> eigen_result(result.vector, ND);
	eigen_result=VectorXd::Zero(ND);

	// TODO check parallel with accumulating on the sum
	// TODO this can be done at the same time as computing the lower right submatrix
	//      to avoid re-computing the kernel hessian
#pragma omp for
	for (auto ind1=0; ind1<ND; ind1++)
		for (auto ind2=0; ind2<ND; ind2++)
		{
			auto ai = idx_to_ai(ind1);
			auto a = ai.first;
			auto i = ai.second;
			auto bj = idx_to_ai(ind2);
			auto b = bj.first;
			auto j = bj.second;
			auto entry = kernel_hessian_component(a, b, i, j);
			result[ind1] += h[ind2] * entry;
		}

	eigen_result /= N;
	eigen_result += m_lambda * eigen_h;
	return result;
}

std::pair<SGMatrix<float64_t>, SGVector<float64_t>> KernelExpFamilyNystromImpl::build_system_slow_low_memory() const
{
	// TODO benchmark against build_system of full estimator
	auto D = get_num_dimensions();
	auto N = get_num_data_lhs();
	auto ND = N*D;
	auto m = get_num_rkhs_basis();

	SG_SINFO("Allocating memory for system.\n");
	SGMatrix<float64_t> A(ND+1, m+1);
	Map<MatrixXd> eigen_A(A.matrix, ND+1, m+1);
	SGVector<float64_t> b(ND+1);
	Map<VectorXd> eigen_b(b.vector, ND+1);

	// TODO all this can be done in a single pass over the data
	// TODO should have an option to store the kernel hessians to speed things up when possible
	// TODO think of a block scheme where one kernel hessian (or N) are stored and less things are recomputed?

	SG_SINFO("Computing h.\n");
	auto h = compute_h();
	auto eigen_h=Map<VectorXd>(h.vector, ND);

	SG_SINFO("Computing xi norm.\n");
	auto xi_norm_2 = compute_xi_norm_2();

	SG_SINFO("Populating A matrix.\n");
	// A[0, 0] = np.dot(h, h) / n + lmbda * xi_norm_2
	A(0,0) = eigen_h.squaredNorm() / N + m_lambda * xi_norm_2;

	// TODO parallelise properly, read up on openmp
	// A_mn[1 + row_idx, 1 + col_idx] = compute_lower_right_submatrix_component(X, lmbda, inds[row_idx], col_idx, sigma)
#pragma omp parallel for
	for (auto col_idx=0; col_idx<m; col_idx++)
	{
		for (auto row_idx=0; row_idx<ND; row_idx++)
			A(1+row_idx, 1+col_idx) = compute_lower_right_submatrix_element(row_idx, m_inds[col_idx]);
	}

	// A_mn[0, 1:] = compute_first_row_without_storing(X, h, N, lmbda, sigma)
	auto first_row = compute_first_row_no_storing();
	Map<VectorXd> eigen_first_row(first_row.vector, ND);
	eigen_A.col(0).segment(1, ND) = eigen_first_row;

	// A_mn[1:, 0] = A_mn[0, inds + 1]
	for (auto ind_idx=0; ind_idx<m; ind_idx++)
		eigen_A(0,ind_idx+1) = first_row[m_inds[ind_idx]];

	// b[0] = -xi_norm_2; b[1:] = -h.reshape(-1)
	b[0] = -xi_norm_2;
	eigen_b.segment(1, ND) = -eigen_h;

	return std::pair<SGMatrix<float64_t>, SGVector<float64_t>>(A, b);
}

std::pair<SGMatrix<float64_t>, SGVector<float64_t>> KernelExpFamilyNystromImpl::build_system_fast_high_memory() const
{
	// TODO actually implement in the fast mem hungry way

	// TODO benchmark against build_system of full estimator
	auto D = get_num_dimensions();
	auto N = get_num_data_lhs();
	auto ND = N*D;
	auto m = get_num_rkhs_basis();

	SG_SINFO("Allocating memory for system.\n");
	SGMatrix<float64_t> A(ND+1, m+1);
	Map<MatrixXd> eigen_A(A.matrix, ND+1, m+1);
	SGVector<float64_t> b(ND+1);
	Map<VectorXd> eigen_b(b.vector, ND+1);

	// TODO all this can be done in a single pass over the data
	// TODO should have an option to store the kernel hessians to speed things up when possible
	// TODO think of a block scheme where one kernel hessian (or N) are stored and less things are recomputed?

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
	auto eigen_col_sub_sampled_hessian = Map<MatrixXd>(col_sub_sampled_hessian.matrix, ND, m);

	for (auto j=0; j<m; j++)
	{
		memcpy(col_sub_sampled_hessian.get_column_vector(j),
				all_hessians.get_column_vector(m_inds[j]),
				sizeof(float64_t)*ND);
	}

	SG_SINFO("Populating A matrix.\n");
	// A[0, 0] = np.dot(h, h) / n + lmbda * xi_norm_2
	A(0,0) = eigen_h.squaredNorm() / N + m_lambda * xi_norm_2;

	// can use noalias to speed up as matrices are definitely different
	eigen_A.block(1,1,ND,m).noalias()=eigen_hessians*eigen_col_sub_sampled_hessian / N + m_lambda*eigen_col_sub_sampled_hessian;
	// A_mn[0, 1:] = compute_first_row(h, all_hessians, n, lmbda)
	eigen_A.col(0).segment(1, ND).noalias() = eigen_hessians*eigen_h / N + m_lambda*eigen_h;

	// A_mn[1:, 0] = A_mn[0, inds + 1]
	for (auto ind_idx=0; ind_idx<m; ind_idx++)
		eigen_A(0, ind_idx+1) = eigen_A(m_inds[ind_idx]+1, 0);

	// b[0] = -xi_norm_2; b[1:] = -h.reshape(-1)
	b[0] = -xi_norm_2;
	eigen_b.segment(1, ND) = -eigen_h;

	return std::pair<SGMatrix<float64_t>, SGVector<float64_t>>(A, b);
}

void KernelExpFamilyNystromImpl::fit()
{
	auto D = get_num_dimensions();
	auto N = get_num_data_lhs();
	auto ND = N*D;
	auto m = get_num_rkhs_basis();

	SG_SINFO("Building system.\n");
	std::pair<SGMatrix<float64_t>, SGVector<float64_t>> A_nm_b;
	if (m_low_memory_mode)
		A_nm_b = build_system_slow_low_memory();
	else
		A_nm_b = build_system_fast_high_memory();

	auto eigen_A_nm = Map<MatrixXd>(A_nm_b.first.matrix, ND+1, m+1);
	auto eigen_b = Map<VectorXd>(A_nm_b.second.vector, ND+1);

	SGMatrix<float64_t> A(m+1,m+1);
	Map<MatrixXd> eigen_A(A.matrix, m+1, m+1);

	SG_SINFO("Solving system.\n");
	eigen_A = eigen_A_nm.transpose()*eigen_A_nm;
	auto b_m = eigen_A_nm.transpose()*eigen_b;

	m_alpha_beta = SGVector<float64_t>(m+1);
	auto eigen_alpha_beta = Map<VectorXd>(m_alpha_beta.vector, m+1);
	auto A_pinv = pinv(A);
	Map<MatrixXd> eigen_pinv(A_pinv.matrix, A_pinv.num_rows, A_pinv.num_cols);

	eigen_alpha_beta = eigen_pinv*b_m;
}

SGMatrix<float64_t> KernelExpFamilyNystromImpl::pinv(const SGMatrix<float64_t>& A)
{
	// based on the snippet from
	// http://eigen.tuxfamily.org/index.php?title=FAQ#Is_there_a_method_to_compute_the_.28Moore-Penrose.29_pseudo_inverse_.3F
	auto eigen_A=Map<MatrixXd>(A.matrix, A.num_rows, A.num_cols);

	JacobiSVD<MatrixXd> svd(eigen_A, ComputeThinU | ComputeThinV);
	auto singular_values = svd.singularValues();
	auto V=svd.matrixV();
	auto U=svd.matrixU();

	// tol = eps⋅max(m,n) * max(singularvalues)
	// this is done in numpy/Octave & co
	// c.f. https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse#Singular_value_decomposition_.28SVD.29
	float64_t pinv_tol = CMath::MACHINE_EPSILON * CMath::max(A.num_rows, A.num_cols) * singular_values.maxCoeff();

	VectorXd inv_singular_values(singular_values.rows());
	for (auto i=0; i<singular_values.rows(); i++)
	{
		if (singular_values(i) > pinv_tol)
			inv_singular_values(i)=1.0/singular_values(i);
		else
			inv_singular_values(i)=0;
	}

	SGMatrix<float64_t> A_pinv(A.num_cols, A.num_rows);
	Map<MatrixXd> eigen_pinv(A_pinv.matrix, A_pinv.num_rows, A_pinv.num_cols);
	eigen_pinv = (V*inv_singular_values.asDiagonal()*U.transpose());

	return A_pinv;
}

float64_t KernelExpFamilyNystromImpl::kernel_dx_component(const SGVector<float64_t>& a, index_t idx_b, index_t i)
{
	auto D = get_num_dimensions();
	Map<VectorXd> eigen_a(a.vector, D);
	Map<VectorXd> eigen_b(m_data_lhs.get_column_vector(idx_b), D);

	//k = gaussian_kernel(x_2d, y_2d, sigma)
	auto diff = eigen_b-eigen_a;
	auto k=CMath::exp(-diff.squaredNorm() / m_sigma);

	return 2*k*diff[i]/m_sigma;
}

float64_t KernelExpFamilyNystromImpl::kernel_dx_component(index_t idx_a, index_t idx_b, index_t i)
{
	//k = gaussian_kernel(x_2d, y_2d, sigma)
	auto diff_i = difference_component(idx_a, idx_b, i);
	auto k=kernel(idx_a, idx_b);

	return 2*k*diff_i/m_sigma;
}

float64_t KernelExpFamilyNystromImpl::kernel_dx_dx_component(const SGVector<float64_t>& a, index_t idx_b, index_t i)
{
	auto D = get_num_dimensions();
	Map<VectorXd> eigen_a(a.vector, D);
	Map<VectorXd> eigen_b(m_data_lhs.get_column_vector(idx_b), D);
	auto sq_diff = (eigen_a-eigen_b).array().pow(2);

	auto k=CMath::exp(-sq_diff.sum() / m_sigma);

	return k*(sq_diff[i]*pow(2.0/m_sigma, 2) -2.0/m_sigma);
}

float64_t KernelExpFamilyNystromImpl::kernel_dx_dx_component(index_t idx_a, index_t idx_b, index_t i)
{
	auto diff_i = difference_component(idx_a, idx_b, i);
	auto k=kernel(idx_a, idx_b);

	return k*(pow(diff_i,2)*pow(2.0/m_sigma, 2) -2.0/m_sigma);
}

SGVector<float64_t> KernelExpFamilyNystromImpl::kernel_dx_i_dx_i_dx_j_component(const SGVector<float64_t>& a, index_t idx_b, index_t i)
{
	auto D = get_num_dimensions();
	Map<VectorXd> eigen_a(a.vector, D);
	Map<VectorXd> eigen_b(m_data_lhs.get_column_vector(idx_b), D);
	auto diff = eigen_b-eigen_a;
	auto sq_diff = diff.array().pow(2).matrix();
	auto k=CMath::exp(-sq_diff.sum() / m_sigma);

	SGVector<float64_t> result(D);
	Map<VectorXd> eigen_result(result.vector, D);

	eigen_result = sq_diff[i]*diff.transpose();
	eigen_result *= k* pow(2.0/m_sigma, 3);
	eigen_result -= k * diff.transpose() * pow(2.0/m_sigma, 2);
	eigen_result[i] -= 2* k * diff[i] * pow(2.0/m_sigma, 2);

	return result;
}

SGVector<float64_t> KernelExpFamilyNystromImpl::kernel_dx_i_dx_i_dx_j_component(index_t idx_a, index_t idx_b, index_t i)
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

SGVector<float64_t> KernelExpFamilyNystromImpl::kernel_dx_i_dx_j_component(const SGVector<float64_t>& a, index_t idx_b, index_t i)
{
	auto D = get_num_dimensions();
	Map<VectorXd> eigen_a(a.vector, D);
	Map<VectorXd> eigen_b(m_data_lhs.get_column_vector(idx_b), D);
	auto diff = eigen_b-eigen_a;
	auto k=CMath::exp(-diff.array().pow(2).sum() / m_sigma);

	SGVector<float64_t> result(D);
	Map<VectorXd> eigen_result(result.vector, D);

	eigen_result = diff[i]*diff;
	eigen_result *= k * pow(2.0/m_sigma, 2);
	eigen_result[i] -= k * 2.0/m_sigma;

	return result;
}

SGVector<float64_t> KernelExpFamilyNystromImpl::kernel_dx_i_dx_j_component(index_t idx_a, index_t idx_b, index_t i)
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

index_t KernelExpFamilyNystromImpl::get_num_rkhs_basis() const
{
	return m_inds.vlen;
}

float64_t KernelExpFamilyNystromImpl::log_pdf(const SGVector<float64_t>& x)
{
	auto N = get_num_data_lhs();
	auto m = get_num_rkhs_basis();

	float64_t xi = 0;
	float64_t beta_sum = 0;

	for (auto ind_idx=0; ind_idx<m; ind_idx++)
	{
		auto ai = idx_to_ai(m_inds[ind_idx]);
		auto a = ai.first;
		auto i = ai.second;

		auto grad_x_xa_i = kernel_dx_component(x, a, i);
		auto xi_grad_i = kernel_dx_dx_component(x, a, i);

		xi += xi_grad_i;
		beta_sum += grad_x_xa_i * m_alpha_beta[1+ind_idx];
	}

	return m_alpha_beta[0]*xi/N + beta_sum;
}

SGVector<float64_t> KernelExpFamilyNystromImpl::grad(const SGVector<float64_t>& x)
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
		auto ai = idx_to_ai(m_inds[ind_idx]);
		auto a = ai.first;
		auto i = ai.second;

		auto xi_gradient_mat_component = kernel_dx_i_dx_i_dx_j_component(x, a, i);
		Map<VectorXd> eigen_xi_gradient_mat_component(xi_gradient_mat_component.vector, D);
		auto left_arg_hessian_component = kernel_dx_i_dx_j_component(x, a, i);
		Map<VectorXd> eigen_left_arg_hessian_component(left_arg_hessian_component.vector, D);

		eigen_xi_grad += eigen_xi_gradient_mat_component;
		eigen_beta_sum_grad += eigen_left_arg_hessian_component * m_alpha_beta[1+ind_idx];
	}

	// re-use memory
	eigen_xi_grad *= m_alpha_beta[0] / N;
	eigen_xi_grad += eigen_beta_sum_grad;
	return xi_grad;
}
