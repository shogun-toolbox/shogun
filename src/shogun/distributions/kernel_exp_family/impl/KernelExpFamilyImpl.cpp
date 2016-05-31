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
#include <shogun/io/SGIO.h>

#include "KernelExpFamilyImpl.h"

using namespace shogun;
using namespace Eigen;

index_t KernelExpFamilyImpl::get_num_dimensions() const
{
	return m_data_lhs.num_rows;
}

index_t KernelExpFamilyImpl::get_num_data_lhs() const
{
	return m_data_lhs.num_cols;
}

void KernelExpFamilyImpl::set_test_data(SGMatrix<float64_t> X)
{
	m_data_rhs = X;
	precompute();
}

void KernelExpFamilyImpl::set_test_data(SGVector<float64_t> x)
{
	set_test_data(SGMatrix<float64_t>(x));
}

index_t KernelExpFamilyImpl::get_num_data_rhs() const
{
	return m_data_rhs.num_cols;
}

void KernelExpFamilyImpl::precompute()
{
	// remove potentially previously precomputed quantities to make calls below
	// not use existing matrices
	m_sq_difference_norms = SGMatrix<float64_t>();
	m_differences = SGMatrix<float64_t>();

	SGMatrix<float64_t> sq_difference_norms;
	SGMatrix<float64_t> differences;

	auto D = get_num_dimensions();

	// distinguish symmetric and non-symmetric case
	if (!m_data_rhs.matrix)
	{
		auto N = get_num_data_lhs();
		SG_SINFO("Precomputing symmetric case with N=%d.\n", N);

		// TODO exploit symmetry in storage
		sq_difference_norms = SGMatrix<float64_t>(N,N);
		differences = SGMatrix<float64_t>(D,N*N);

#pragma omp parallel for
		for (auto i=0; i<N; i++)
		{
			for (auto j=0; j<i; j++)
			{
				SGVector<float64_t> diff(differences.get_column_vector(i*N+j), D, false);
				difference(i, j, diff);

				// use symmetry and only remember sign flip
				auto p = differences.get_column_vector(j*N+i);
				memcpy(p, diff.vector, sizeof(float64_t)*D);
				Map<VectorXd> diff2(p, D);
				diff2*=-1;

				sq_difference_norms(i,j)=sq_difference_norm(diff);
				sq_difference_norms(j,i)=sq_difference_norms(i,j);
			}

			// avoid computing distances for equal case
			memset(differences.get_column_vector(i*N+i), 0, sizeof(float64_t)*D);
			sq_difference_norms(i,i)=0;
		}
	}
	else
	// non symmetric case
	{
		auto N_lhs = get_num_data_lhs();
		auto N_rhs = get_num_data_rhs();
		SG_SINFO("Precomputing non symmetric case with N_lhs=%d, N_rhs=%d.\n",
				N_lhs, N_rhs);
		sq_difference_norms = SGMatrix<float64_t>(N_lhs, N_rhs);
		differences = SGMatrix<float64_t>(D,N_lhs*N_rhs);

#pragma omp parallel for
		for (auto i=0; i<N_lhs; i++)
		{
			for (auto j=0; j<N_rhs; j++)
			{
				SGVector<float64_t> diff(differences.get_column_vector(i*N_rhs+j), D, false);
				difference(i, j, diff);

				sq_difference_norms(i,j)=sq_difference_norm(diff);
			}
		}
	}

	// might affect methods, so only set now
	m_differences = differences;
	m_sq_difference_norms = sq_difference_norms;
}

KernelExpFamilyImpl::KernelExpFamilyImpl(SGMatrix<float64_t> data, float64_t sigma, float64_t lambda)
{
	m_data_lhs = data;

	m_sigma = sigma;
	m_lambda = lambda;

	SG_SINFO("Problem size is N=%d, D=%d.\n", get_num_data_lhs(), get_num_dimensions());

	precompute();
}

float64_t KernelExpFamilyImpl::kernel(index_t idx_a, index_t idx_b) const
{
	return CMath::exp(-sq_difference_norm(idx_a, idx_b) / m_sigma);
}

float64_t KernelExpFamilyImpl::sq_difference_norm(index_t idx_a,  index_t idx_b) const
{
	if (m_sq_difference_norms.matrix)
		return m_sq_difference_norms(idx_a, idx_b);

	SGVector<float64_t> diff = difference(idx_a, idx_b);
	return sq_difference_norm(diff);
}

float64_t KernelExpFamilyImpl::sq_difference_norm(const SGVector<float64_t>& diff) const
{
	Map<VectorXd> eigen_diff(diff.vector, diff.vlen);
	return eigen_diff.squaredNorm();
}

SGVector<float64_t> KernelExpFamilyImpl::difference(index_t idx_a, index_t idx_b) const
{
	auto D = get_num_dimensions();

	if (m_differences.matrix)
	{
		auto N_rhs = m_data_rhs.matrix ? get_num_data_rhs(): get_num_data_lhs();
		return SGVector<float64_t>(m_differences.get_column_vector(idx_a*N_rhs+idx_b), D, false);
	}

	SGVector<float64_t> result(D);
	difference(idx_a, idx_b, result);
	return result;
}

void KernelExpFamilyImpl::difference(index_t idx_a, index_t idx_b,
		SGVector<float64_t>& result) const
{
	auto D = get_num_dimensions();
	if (m_differences.matrix)
	{
		auto N_rhs = m_data_rhs.matrix ? get_num_data_rhs() : get_num_data_lhs();
		memcpy(result.vector,
				m_differences.get_column_vector(idx_a*N_rhs+idx_b),
				sizeof(float64_t)*D);
	}
	else
	{
		Map<VectorXd> x(m_data_lhs.get_column_vector(idx_a), D);
		float64_t* right_pointer = m_data_rhs.matrix ?
				m_data_rhs.get_column_vector(idx_b) : m_data_lhs.get_column_vector(idx_b);
		Map<VectorXd> y(right_pointer, D);

		Map<VectorXd> eigen_diff(result.vector, D);
		eigen_diff = y-x;
	}
}

SGMatrix<float64_t> KernelExpFamilyImpl::kernel_dx_dx_dy(index_t idx_a, index_t idx_b) const
{
	auto D = get_num_dimensions();

	SGVector<float64_t> diff=difference(idx_a, idx_b);
	Map<VectorXd> eigen_diff(diff.vector, D);

	auto k=kernel(idx_a,idx_b);

	SGMatrix<float64_t> result(D, D);
	Map<MatrixXd> eigen_result(result.matrix, D, D);
	eigen_result = pow(2./m_sigma,3) * k *
			((eigen_diff.array().pow(2).matrix())*(-eigen_diff.transpose()));
	eigen_result += pow(2./m_sigma,2) * k * 2* eigen_diff.asDiagonal();
	eigen_result.rowwise() +=  (pow(2./m_sigma,2) * k * eigen_diff).transpose();

	return result;
}

float64_t KernelExpFamilyImpl::kernel_dx_dx_dy_dy_sum(index_t idx_a, index_t idx_b) const
{
	auto D = get_num_dimensions();
	SGVector<float64_t> diff=difference(idx_a, idx_b);
	Map<VectorXd> eigen_diff(diff.vector, D);
	auto diff2 = eigen_diff.array().pow(2).matrix();

	//k = gaussian_kernel(x_2d, y_2d, sigma)
	auto k=kernel(idx_a,idx_b);
	auto factor = k*pow(2.0/m_sigma, 3);

	float64_t sum = 0;
	for (auto i=0; i<D; i++)
		for (auto j=0; j<D; j++)
		{
			sum += k*pow(2.0/m_sigma, 4) * (diff2[i]*diff2[j]);
			sum -= factor*(diff2[i]+diff2[j] - 1);
			if (i==j)
				sum -= 4*factor*diff2[i] - 2*factor;
		}

	return sum;
}

SGMatrix<float64_t> KernelExpFamilyImpl::kernel_dx_dx_dy_dy(index_t idx_a, index_t idx_b) const
{
	auto D = get_num_dimensions();
	SGVector<float64_t> diff=difference(idx_a, idx_b);
	Map<VectorXd> eigen_diff(diff.vector, D);
	VectorXd diff2 = eigen_diff.array().pow(2).matrix();

	//k = gaussian_kernel(x_2d, y_2d, sigma)
	auto k=kernel(idx_a,idx_b);

	auto factor = k*pow(2.0/m_sigma, 3);
	SGMatrix<float64_t> result(D, D);
	Map<MatrixXd> eigen_result(result.matrix, D, D);

	//term1 = k * np.outer((x - y), (x - y)) ** 2 * (2.0/sigma)**4
	eigen_result = k*pow(2.0/m_sigma, 4) * (diff2*diff2.transpose());

	//term2 = k * 6 * np.diag((x - y) ** 2) * (2.0/sigma)**3  # diagonal (x-y)
	eigen_result.diagonal() -= 6*factor*diff2;

	//term3 = (1 - np.eye(d)) * k * np.tile((x - y), [d, 1]).T ** 2 * (2.0/sigma)**3  # (x_i-y_i)^2 off-diagonal
	diff2 *= factor;
	eigen_result.rowwise() -=  diff2.transpose();
	eigen_result.colwise() -=  diff2;
	eigen_result.diagonal() += 2*diff2;

	// term5 = k * (1 + 2 * np.eye(d)) * (2.0/sigma)**2
	factor = k*pow(2.0/m_sigma, 2);
	eigen_result.diagonal().array() += 2*factor;
	eigen_result.array() += factor;

	// return term1 - term2 - term3 - term3.T + term5
	return result;
}

SGMatrix<float64_t> KernelExpFamilyImpl::kernel_hessian(index_t idx_a, index_t idx_b) const
{
	auto D = get_num_dimensions();
	SGVector<float64_t> diff=difference(idx_a, idx_b);
	Map<VectorXd> eigen_diff(diff.vector, D);

	//k = gaussian_kernel(x_2d, y_2d, sigma)
	auto k=kernel(idx_a,idx_b);

	SGMatrix<float64_t> result(D, D);
	Map<MatrixXd> eigen_result(result.matrix, D, D);

	// H = k*(2*np.eye(d)/sigma - 4*np.outer(differences, differences)/sigma**2)
	eigen_result = -eigen_diff*eigen_diff.transpose() / pow(m_sigma, 2) * k * 4;
	eigen_result.diagonal().array() += 2*k/m_sigma;

	return result;
}

SGVector<float64_t> KernelExpFamilyImpl::kernel_dx_dx(index_t idx_a, index_t idx_b) const
{
	auto D = get_num_dimensions();
	auto diff = difference(idx_a, idx_b);
	auto eigen_diff = Map<VectorXd>(diff.vector, D);
	auto sq_diff = eigen_diff.array().pow(2);

	auto k=kernel(idx_a, idx_b);

	SGVector<float64_t> result(D);
	Map<VectorXd> eigen_result(result.vector, D);

	// k.T * (sq_differences*(2.0 / sigma)**2 - 2.0/sigma)
	eigen_result = k*(sq_diff*pow(2.0/m_sigma, 2) -2.0/m_sigma);

	return result;
}

SGMatrix<float64_t> KernelExpFamilyImpl::kernel_hessian_all() const
{
	auto D = get_num_dimensions();
	auto N = get_num_data_lhs();
	auto ND = N*D;
	SGMatrix<float64_t> result(ND,ND);
	Map<MatrixXd> eigen_result(result.matrix, ND,ND);

	//TODO exploit symmetry both in computation and storage
#pragma omp parallel for
	for (auto idx_a=0; idx_a<N; idx_a++)
		for (auto idx_b=0; idx_b<N; idx_b++)
		{
			auto r_start = idx_a*D;
			auto c_start = idx_b*D;
			SGMatrix<float64_t> h=kernel_hessian(idx_a, idx_b);
			eigen_result.block(r_start, c_start, D, D) = Map<MatrixXd>(h.matrix, D, D);
			eigen_result.block(c_start, r_start, D, D) = eigen_result.block(r_start, c_start, D, D);
		}

	return result;
}

SGVector<float64_t> KernelExpFamilyImpl::kernel_dx(index_t idx_a, index_t idx_b) const
{
	auto D = get_num_dimensions();

	//k = gaussian_kernel(x_2d, y_2d, sigma)
	auto diff = difference(idx_a, idx_b);

	auto eigen_diff = Map<VectorXd>(diff, D);
	auto k = kernel(idx_a, idx_b);

	SGVector<float64_t> gradient(D);
	Map<VectorXd> eigen_gradient(gradient.vector, D);
	eigen_gradient = 2*k*eigen_diff/m_sigma;
	return gradient;
}

SGVector<float64_t> KernelExpFamilyImpl::compute_h() const
{
	auto D = get_num_dimensions();
	auto N = get_num_data_lhs();
	auto ND = N*D;
	SGVector<float64_t> h(ND);
	Map<VectorXd> eigen_h(h.vector, ND);
	eigen_h = VectorXd::Zero(ND);

#pragma omp parallel for
	for (auto idx_b=0; idx_b<N; idx_b++)
		for (auto idx_a=0; idx_a<N; idx_a++)
		{
			// TODO optimise, no need to store matrix
			SGMatrix<float64_t> temp = kernel_dx_dx_dy(idx_a, idx_b);
			eigen_h.segment(idx_b*D, D) += Map<MatrixXd>(temp.matrix, D,D).colwise().sum();
		}

	eigen_h /= N;

	return h;
}

float64_t KernelExpFamilyImpl::compute_xi_norm_2() const
{
	auto N = get_num_data_lhs();
	float64_t xi_norm_2=0;

#pragma omp parallel for reduction (+:xi_norm_2)
	for (auto idx_a=0; idx_a<N; idx_a++)
		for (auto idx_b=0; idx_b<N; idx_b++)
			xi_norm_2 += kernel_dx_dx_dy_dy_sum(idx_a, idx_b);

	xi_norm_2 /= (N*N);

	return xi_norm_2;
}

std::pair<SGMatrix<float64_t>, SGVector<float64_t>> KernelExpFamilyImpl::build_system() const
{
	auto D = get_num_dimensions();
	auto N = get_num_data_lhs();
	auto ND = N*D;

	// TODO A matrix should be stored exploiting symmetry
	SG_SINFO("Allocating memory for system.\n");
	SGMatrix<float64_t> A(ND+1,ND+1);
	Map<MatrixXd> eigen_A(A.matrix, ND+1,ND+1);
	SGVector<float64_t> b(ND+1);
	Map<VectorXd> eigen_b(b.vector, ND+1);

	// TODO all this can be done using a single pass over all data

	SG_SINFO("Computing h.\n");
	auto h = compute_h();
	auto eigen_h=Map<VectorXd>(h.vector, ND);

	SG_SINFO("Computing all kernel Hessians.\n");
	auto all_hessians = kernel_hessian_all();
	auto eigen_all_hessians = Map<MatrixXd>(all_hessians.matrix, ND, ND);

	SG_SINFO("Computing xi norm.\n");
	auto xi_norm_2 = compute_xi_norm_2();

	SG_SINFO("Populating A matrix.\n");
	// A[0, 0] = np.dot(h, h) / n + lmbda * xi_norm_2
	A(0,0) = eigen_h.squaredNorm() / N + m_lambda * xi_norm_2;

	// A[1:, 1:] = np.dot(all_hessians, all_hessians) / N + lmbda * all_hessians
	// A[0, 1:] = np.dot(h, all_hessians) / n + lmbda * h; A[1:, 0] = A[0, 1:]
	// can use noalias to speed up as matrices are definitely different
	eigen_A.block(1,1,ND,ND).noalias()=eigen_all_hessians*eigen_all_hessians / N + m_lambda*eigen_all_hessians;
	eigen_A.row(0).segment(1, ND).noalias() = eigen_all_hessians*eigen_h / N + m_lambda*eigen_h;
	eigen_A.col(0).segment(1, ND).noalias() = eigen_A.row(0).segment(1, ND);

	// b[0] = -xi_norm_2; b[1:] = -h.reshape(-1)
	b[0] = -xi_norm_2;
	eigen_b.segment(1, ND) = -eigen_h;

	return std::pair<SGMatrix<float64_t>, SGVector<float64_t>>(A, b);
}

void KernelExpFamilyImpl::fit()
{
	auto D = get_num_dimensions();
	auto N = get_num_data_lhs();
	auto ND = N*D;

	SG_SINFO("Building system.\n");
	auto A_b = build_system();
	auto eigen_A = Map<MatrixXd>(A_b.first.matrix, ND+1, ND+1);
	auto eigen_b = Map<VectorXd>(A_b.second.vector, ND+1);

	m_alpha_beta = SGVector<float64_t>(ND+1);
	auto eigen_alpha_beta = Map<VectorXd>(m_alpha_beta.vector, ND+1);

	SG_SINFO("Solving system.\n");
	eigen_alpha_beta = eigen_A.ldlt().solve(eigen_b);
}

SGVector<float64_t> KernelExpFamilyImpl::log_pdf(const SGMatrix<float64_t> X)
{
	set_test_data(X);
	auto N_test = get_num_data_rhs();
	SGVector<float64_t> result(N_test);
#pragma omp parallel for
	for (auto i=0; i<N_test; ++i)
		result[i] = log_pdf(i);

	return result;
}

float64_t KernelExpFamilyImpl::log_pdf(SGVector<float64_t> x)
{
	set_test_data(x);
	return log_pdf(0);
}

float64_t KernelExpFamilyImpl::log_pdf(index_t idx_test) const
{
	auto D = get_num_dimensions();
	auto N = get_num_data_lhs();

	float64_t xi = 0;
	float64_t beta_sum = 0;

	Map<VectorXd> eigen_alpha_beta(m_alpha_beta.vector, N*D+1);
	for (auto idx_a=0; idx_a<N; idx_a++)
	{
		SGVector<float64_t> k=kernel_dx_dx(idx_a, idx_test);
		Map<VectorXd> eigen_k(k.vector, D);
		xi += eigen_k.sum() / N;

		auto grad_x_xa = kernel_dx(idx_a, idx_test);
		Map<VectorXd> eigen_grad_x_xa(grad_x_xa.vector, D);

		// betasum += np.dot(gradient_x_xa, beta[a, :])
		// note: sign flip as different argument order compared to Python code
		beta_sum -= eigen_grad_x_xa.transpose()*eigen_alpha_beta.segment(1+idx_a*D, D);

	}
	return m_alpha_beta[0]*xi + beta_sum;
}

SGVector<float64_t> KernelExpFamilyImpl::grad(SGVector<float64_t> x)
{
	set_test_data(x);
	return grad(0);
}

SGVector<float64_t> KernelExpFamilyImpl::grad(index_t idx_test)
{
	auto D = get_num_dimensions();
	auto N = get_num_data_lhs();

	SGVector<float64_t> xi_grad(D);
	SGVector<float64_t> beta_sum_grad(D);
	Map<VectorXd> eigen_xi_grad(xi_grad.vector, D);
	Map<VectorXd> eigen_beta_sum_grad(beta_sum_grad.vector, D);
	eigen_xi_grad = VectorXd::Zero(D);
	eigen_beta_sum_grad.array() = VectorXd::Zero(D);

	Map<VectorXd> eigen_alpha_beta(m_alpha_beta.vector, N*D+1);
	for (auto a=0; a<N; a++)
	{
		SGMatrix<float64_t> g=kernel_dx_i_dx_i_dx_j(a, idx_test);
		Map<MatrixXd> eigen_g(g.matrix, D, D);
		eigen_xi_grad -= eigen_g.colwise().sum();

		// left_arg_hessian = gaussian_kernel_dx_i_dx_j(x, x_a, sigma)
		// betasum_grad += beta[a, :].dot(left_arg_hessian)
		// TODO storage is not necessary here
		// note: sign flip as different argument order compared to Python code
		auto left_arg_hessian = kernel_dx_i_dx_j(a, idx_test);
		Map<MatrixXd> eigen_left_arg_hessian(left_arg_hessian.matrix, D, D);
		eigen_beta_sum_grad += eigen_left_arg_hessian*eigen_alpha_beta.segment(1+a*D, D).matrix();
	}

	// return alpha * xi_grad + betasum_grad
	eigen_xi_grad *= m_alpha_beta[0] / N;
	return xi_grad + beta_sum_grad;
}

SGMatrix<float64_t> KernelExpFamilyImpl::kernel_dx_i_dx_i_dx_j(index_t idx_a, index_t idx_b) const
{
	auto D = get_num_dimensions();
	auto diff = difference(idx_a, idx_b);
	auto eigen_diff = Map<VectorXd>(diff.vector, D);
	auto sq_diff = eigen_diff.array().pow(2).matrix();
	auto k = kernel(idx_a, idx_b);

	SGMatrix<float64_t> result(D, D);
	Map<MatrixXd> eigen_result(result.matrix, D, D);

	// pairwise_dist_squared_i = np.outer((y-x)**2, y-x)
	// term1 = k*pairwise_dist_squared_i * (2.0/sigma)**3
	eigen_result = sq_diff*eigen_diff.transpose();
	eigen_result *= k* pow(2.0/m_sigma, 3);

	// row_repeated_distances = np.tile(y-x, [d,1])
	// term2 = k*row_repeated_distances * (2.0/sigma)**2
	eigen_result.rowwise() -= k * eigen_diff.transpose() * pow(2.0/m_sigma, 2);

	// term3 = term2*2*np.eye(d)
	eigen_result.diagonal() -= 2* k * eigen_diff * pow(2.0/m_sigma, 2);

	// return term1 - term2 - term3
	return result;
}

SGMatrix<float64_t> KernelExpFamilyImpl::kernel_dx_i_dx_j(index_t idx_a, index_t idx_b) const
{
	auto D = get_num_dimensions();
	auto diff = difference(idx_a, idx_b);
	auto eigen_diff = Map<VectorXd>(diff.vector, D);
	auto k=kernel(idx_a, idx_b);

	SGMatrix<float64_t> result(D, D);
	Map<MatrixXd> eigen_result(result.matrix, D, D);

	// pairwise_dist = np.outer(y-x, y-x)
	// term1 = k*pairwise_dist * (2.0/sigma)**2
	eigen_result = eigen_diff*eigen_diff.transpose();
	eigen_result *= k * pow(2.0/m_sigma, 2);

	// term2 = k*np.eye(d) * (2.0/sigma)
	eigen_result.diagonal().array() -= k * 2.0/m_sigma;

	// return term1 - term2
	return result;
}
