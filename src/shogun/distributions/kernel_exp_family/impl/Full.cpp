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

#include "Full.h"
#include "kernel/Base.h"

using namespace shogun;
using namespace shogun::kernel_exp_family_impl;
using namespace Eigen;

Full::Full(SGMatrix<float64_t> data,
		std::shared_ptr<kernel::Base> kernel, float64_t lambda, float64_t base_measure_cov_ridge,
		bool init_base_and_data) :
		Base(data, kernel, lambda, init_base_and_data)
{
	m_base_measure_cov_ridge = base_measure_cov_ridge;
}

float64_t Full::base_measure_log_pdf(const SGVector<float64_t>& vec) const
{
	auto D = get_num_dimensions();

	Map<VectorXd> mu(m_mu.vector, D);
	Map<MatrixXd> L(m_Sigma_L.matrix, D, D);
	Map<VectorXd> eigen_vec(vec.vector, D);

	// a=Sigma^{-1}*(x-mu) via 2 triangular solves
	VectorXd centered_x = eigen_vec - mu;
	VectorXd a=L.triangularView<Lower>().solve(centered_x);
	a=L.triangularView<Lower>().adjoint().solve(a);

	auto result = -0.5 * centered_x.dot(a);

	return result;
}

SGVector<float64_t> Full::base_measure_dx(const SGVector<float64_t>& vec) const
{
	auto D = get_num_dimensions();

	Map<VectorXd> mu(m_mu.vector, D);
	Map<MatrixXd> L(m_Sigma_L.matrix, D, D);
	Map<VectorXd> eigen_vec(vec.vector, D);

	SGVector<float64_t> result(D);
	Map<VectorXd> eigen_result(result.vector, D);

	// a=Sigma^{-1}*(x-mu) via 2 triangular solves
	VectorXd centered_x = eigen_vec - mu;
	eigen_result=L.triangularView<Lower>().solve(centered_x);
	eigen_result=-L.triangularView<Lower>().adjoint().solve(eigen_result);

	return result;
}

SGMatrix<float64_t> Full::base_measure_dx(const SGMatrix<float64_t>& mat) const
{
	auto D = get_num_dimensions();
	auto N = get_num_data();

	Map<VectorXd> mu(m_mu.vector, D);
	Map<MatrixXd> L(m_Sigma_L.matrix, D, D);

	Map<MatrixXd> eigen_data(m_data.matrix, D, N);

	SGMatrix<float64_t> result(D, N);
	Map<MatrixXd> eigen_result(result.matrix, D, N);

	eigen_result = eigen_data.colwise() - mu;
	eigen_result=L.triangularView<Lower>().solve(eigen_result);
	eigen_result=-L.triangularView<Lower>().adjoint().solve(eigen_result);

	return result;
}

SGVector<float64_t> Full::base_measure_dx_dx_times_vec(const SGVector<float64_t>& vec,
		const SGVector<float64_t>& other) const
{
	auto D = get_num_dimensions();

	Map<VectorXd> mu(m_mu.vector, D);
	Map<MatrixXd> L(m_Sigma_L.matrix, D, D);
	Map<VectorXd> eigen_vec(vec.vector, D);
	Map<VectorXd> eigen_other(other.vector, D);

	SGVector<float64_t> result(D);
	Map<VectorXd> eigen_result(result.vector, D);

	// a=Sigma^{-1}*(x-mu) via 2 triangular solves
	eigen_result=L.triangularView<Lower>().solve(eigen_other);
	eigen_result=-L.triangularView<Lower>().adjoint().solve(eigen_result);

	return result;
}

SGVector<float64_t> Full::compute_h() const
{
	auto D = get_num_dimensions();
	auto N_basis = get_num_basis();
	auto N_data = get_num_data();

	auto system_size = N_basis*D;
	SGVector<float64_t> h(system_size);
	Map<VectorXd> eigen_h(h.vector, system_size);
	eigen_h = VectorXd::Zero(system_size);

#pragma omp parallel for
	for (auto idx_a=0; idx_a<N_basis; idx_a++)
		for (auto idx_b=0; idx_b<N_data; idx_b++)
		{
			// TODO optimise, no need to store matrix
			// TODO optimise, pull allocation out of the loop
			SGMatrix<float64_t> temp = m_kernel->dx_dy_dy(idx_a, idx_b);
			eigen_h.segment(idx_a*D, D) += Map<MatrixXd>(temp.matrix, D,D).colwise().sum();
		}

	eigen_h /= N_data;

	return h;
}

//float64_t Full::compute_xi_norm_2() const
//{
//	auto N = get_num_basis();
//	float64_t xi_norm_2=0;
//
//#pragma omp parallel for reduction (+:xi_norm_2)
//	for (auto idx_a=0; idx_a<N; idx_a++)
//		for (auto idx_b=0; idx_b<N; idx_b++)
//			xi_norm_2 += m_kernel->dx_dx_dy_dy_sum(idx_a, idx_b);
//
//	xi_norm_2 /= (N*N);
//
//	return xi_norm_2;
//}

void Full::fit()
{
	REQUIRE(is_basis_equal_data(), "Cannot proceed with set data. Reset data!\n");

	auto D = get_num_dimensions();
	auto N = get_num_basis(); // same as number of data
	auto system_size = get_system_size();

	if (m_base_measure_cov_ridge)
	{
		SG_SWARNING("TODO: make base measure modular.\n");
		SG_SINFO("Computing covariance of data for base measure.\n");
		m_mu = SGVector<float64_t>(D);
		Map<VectorXd> eigen_mu(m_mu.vector, D);
		m_Sigma_L = SGMatrix<float64_t>(D, D);
		Map<MatrixXd> eigen_Sigma_L(m_Sigma_L.matrix, D, D);

		Map<MatrixXd> eigen_data(m_data.matrix, D, N);
		eigen_mu = eigen_data.rowwise().mean();

		auto centered = eigen_data.colwise() - eigen_mu;
		eigen_Sigma_L = (centered * centered .adjoint()) / (N - 1);
		for (auto i=0; i<D; i++)
			eigen_Sigma_L(i,i)+=m_base_measure_cov_ridge;

		SG_SINFO("Computing Cholesky of covariance for base measure.\n");
		eigen_Sigma_L = eigen_Sigma_L.llt().matrixL();
	}

	SG_SINFO("Computing h.\n");
	auto h = compute_h();
	auto eigen_h=Map<VectorXd>(h.vector, system_size);

	SG_SINFO("Computing all kernel Hessians.\n");
	auto G = m_kernel->dx_dy_all();
	for (auto i=0; i<system_size; i++)
		G(i,i) += N*m_lambda;

	auto eigen_G = Map<MatrixXd>(G.matrix, system_size, system_size);

	if (m_base_measure_cov_ridge)
	{
		SG_SINFO("Computing base measure scores.\n");
		auto q0 = base_measure_dx(m_data);
		Map<VectorXd> eigen_q0(q0.matrix, system_size);
		eigen_h += (eigen_G*eigen_q0)/N;
	}

	m_beta = SGVector<float64_t>(system_size);
	auto eigen_beta = Map<VectorXd>(m_beta.vector, system_size);

	SG_SINFO("Solving with LLT.\n");
	auto solver = LLT<MatrixXd>();
	solver.compute(eigen_G);

	if (!solver.info() == Eigen::Success)
	{
		SG_SERROR("Numerical problems computing LLT. This usually means that"
				" either the regularizer is too small, or the problem is "
				"badly conditioned.\n");
	}

	SG_SINFO("Constructing solution.\n");
	eigen_beta = solver.solve(eigen_h / m_lambda);
}

void Full::log_pdf_xi_add(index_t basis_ind, index_t idx_test, float64_t& xi) const
{
	auto N = get_num_basis();
	auto D = get_num_dimensions();
	SGVector<float64_t> k=m_kernel->dx_dx(basis_ind, idx_test);
	Map<VectorXd> eigen_k(k.vector, D);
	xi += eigen_k.sum() / N;
}

void Full::log_pdf_xi_result(float64_t xi, float64_t& result) const
{
	// note the N normalisation is already in the xi
	result -= 1.0/(m_lambda)*xi;
}

float64_t Full::log_pdf(index_t idx_test) const
{
	auto D = get_num_dimensions();
	auto N = get_num_basis();

	float64_t xi = 0;
	float64_t beta_sum = 0;

	Map<VectorXd> eigen_beta(m_beta.vector, get_system_size());
	for (auto idx_a=0; idx_a<N; idx_a++)
	{
		log_pdf_xi_add(idx_a, idx_test, xi);

		// TODO optimise, compute all gradients before loop in batch mode
		auto grad_x_xa = m_kernel->dx(idx_a, idx_test);
		Map<VectorXd> eigen_grad_xa(grad_x_xa.vector, D);

		// betasum += np.dot(gradient_x_xa, beta[a, :])
		beta_sum += eigen_grad_xa.dot(eigen_beta.segment(idx_a*D, D));

		if (m_base_measure_cov_ridge)
		{
			// TODO precompute all scores for base measure in batch mode
			SGVector<float64_t> vec(m_basis.get_column_vector(idx_a), D, false);
			auto q0_score = base_measure_dx(vec);
			Map<VectorXd> eigen_q0_score(q0_score.vector, D);
			xi += eigen_grad_xa.dot(eigen_q0_score) / N;
		}
	}

	auto result = beta_sum;
	log_pdf_xi_result(xi, result);

	if (m_base_measure_cov_ridge)
	{
		SGVector<float64_t> test_vec(m_data.get_column_vector(idx_test), D, false);
		auto log_q0 = base_measure_log_pdf(test_vec);
		result += log_q0;
	}

	return result;
}

void Full::grad_xi_add(index_t basis_ind, index_t idx_test,
		SGVector<float64_t>& xi_grad) const
{
	auto D = get_num_dimensions();
	Map<VectorXd> eigen_xi_grad(xi_grad.vector, D);
	SGMatrix<float64_t> g=m_kernel->dx_i_dx_i_dx_j(basis_ind, idx_test);
	Map<MatrixXd> eigen_g(g.matrix, D, D);
	eigen_xi_grad -= eigen_g.colwise().sum();
}

void Full::grad_xi_result(const SGVector<float64_t>& xi_grad,
		 SGVector<float64_t>& result) const
{
	auto D = get_num_dimensions();
	auto N = get_num_basis();

	// alpha * xi_grad + betasum_grad
	// note that xi is not yet normalised
	Map<VectorXd> eigen_result(result.vector, D);
	Map<VectorXd> eigen_xi_grad(xi_grad.vector, D);

	eigen_result += -1.0 / (m_lambda*N) * eigen_xi_grad;
}

SGVector<float64_t> Full::grad(index_t idx_test) const
{
	auto D = get_num_dimensions();
	auto N = get_num_basis();

	SGVector<float64_t> xi_grad(D);
	Map<VectorXd> eigen_xi_grad(xi_grad.vector, D);
	SGVector<float64_t> beta_grad_sum(D);
	Map<VectorXd> eigen_beta_grad_sum(beta_grad_sum.vector, D);
	eigen_xi_grad = VectorXd::Zero(D);
	eigen_beta_grad_sum.array() = VectorXd::Zero(D);

	Map<VectorXd> eigen_beta(m_beta.vector, get_system_size());
	for (auto a=0; a<N; a++)
	{
		grad_xi_add(a, idx_test, xi_grad);

		// left_arg_hessian = gaussian_kernel_dx_i_dx_j(x, x_a, sigma)
		// betasum_grad += beta[a, :].dot(left_arg_hessian)
		// TODO storage is not necessary here
		auto left_arg_hessian = m_kernel->dx_i_dx_j(a, idx_test);
		Map<MatrixXd> eigen_left_arg_hessian(left_arg_hessian.matrix, D, D);
		eigen_beta_grad_sum -= eigen_left_arg_hessian*eigen_beta.segment(a*D, D).matrix();

		if (m_base_measure_cov_ridge)
		{
			// TODO precompute all scores for base measure in batch mode
			SGVector<float64_t> vec(m_basis.get_column_vector(a), D, false);
			SGVector<float64_t> q0_score = base_measure_dx(vec);
			Map<VectorXd> eigen_q0_score(q0_score.vector, D);

			eigen_xi_grad += eigen_left_arg_hessian*eigen_q0_score;

			auto left_arg_grad = m_kernel->dx(a, idx_test);
			auto q0_2_times_vec=base_measure_dx_dx_times_vec(vec, left_arg_grad);
			eigen_xi_grad += Map<VectorXd>(q0_2_times_vec.vector, D);
		}
	}

	auto result = beta_grad_sum;
	grad_xi_result(xi_grad, result);

	if (m_base_measure_cov_ridge)
	{
		SGVector<float64_t> test_vec(m_data.get_column_vector(idx_test), D, false);
		auto log_q0_grad = base_measure_dx(test_vec);
		Map<VectorXd> eigen_log_q0_grad(log_q0_grad.vector, D);
		eigen_beta_grad_sum += eigen_log_q0_grad;
	}
	return beta_grad_sum;
}

void Full::hessian_xi_add(index_t basis_ind, index_t idx_test,
		SGMatrix<float64_t>& xi_hessian) const
{
	auto D = get_num_dimensions();
	auto xi_hess_sum = m_kernel->dx_i_dx_j_dx_k_dx_k_row_sum(basis_ind, idx_test);

	Map<MatrixXd> eigen_xi_hessian(xi_hessian.matrix, D, D);
	Map<MatrixXd> eigen_xi_hess_sum(xi_hess_sum.matrix, D, D);
	eigen_xi_hessian += eigen_xi_hess_sum;
}

void Full::hessian_xi_result(const SGMatrix<float64_t>& xi_hessian,
		SGMatrix<float64_t>& result) const
{
	auto D = get_num_dimensions();
	auto N = get_num_basis();

	Map<MatrixXd> eigen_xi_hessian(xi_hessian.matrix, D, D);
	Map<MatrixXd> eigen_result(result.matrix, D, D);

	// note that xi is not yet normalised
	eigen_result += eigen_xi_hessian * -1.0/(m_lambda*N);
}

SGMatrix<float64_t> Full::hessian(index_t idx_test) const
{
	REQUIRE(!m_base_measure_cov_ridge, "Base measure not implemented for Hessian.\n");
	auto N = get_num_basis();
	auto D = get_num_dimensions();

	SGMatrix<float64_t> xi_hessian(D, D);
	Map<MatrixXd> eigen_xi_hessian(xi_hessian.matrix, D, D);
	SGMatrix<float64_t> beta_sum_hessian(D, D);

	Map<MatrixXd> eigen_beta_sum_hessian(beta_sum_hessian.matrix, D, D);

	eigen_xi_hessian = MatrixXd::Zero(D, D);
	eigen_beta_sum_hessian = MatrixXd::Zero(D, D);

	Map<VectorXd> eigen_beta(m_beta.vector, get_system_size());

	for (auto a=0; a<N; a++)
	{
		hessian_xi_add(a, idx_test, xi_hessian);

		SGVector<float64_t> beta_a(eigen_beta.segment(a*D, D).data(), D, false);
		auto beta_hess_sum = m_kernel->dx_i_dx_j_dx_k_dot_vec(a, idx_test, beta_a);
		Map<MatrixXd> eigen_beta_hess_sum(beta_hess_sum.matrix, D, D);
		eigen_beta_sum_hessian += eigen_beta_hess_sum;
	}

	auto result = beta_sum_hessian;
	hessian_xi_result(xi_hessian, result);

	return result;
}

void Full::hessian_diag_xi_add(index_t basis_ind, index_t idx_test,
		SGVector<float64_t>& xi_hessian_diag) const
{
	auto D = get_num_dimensions();
	Map<VectorXd> eigen_xi_hessian_diag(xi_hessian_diag.vector, D);

	for (auto i=0; i<D; i++)
	{
		eigen_xi_hessian_diag[i] += m_kernel->dx_i_dx_j_dx_k_dx_k_row_sum_component(
				basis_ind, idx_test, i, i);
	}
}

void Full::hessian_diag_xi_result(const SGVector<float64_t>& xi_hessian_diag,
		SGVector<float64_t>& result) const
{
	auto D = get_num_dimensions();
	auto N = get_num_basis();

	Map<VectorXd> eigen_xi_hessian_diag(xi_hessian_diag.vector, D);
	Map<VectorXd> eigen_result(result.vector, D);

	// note that xi is not yet normalised
	eigen_result += eigen_xi_hessian_diag * -1.0/(m_lambda*N);
}

SGVector<float64_t> Full::hessian_diag(index_t idx_test) const
{
	REQUIRE(!m_base_measure_cov_ridge, "Base measure not implemented for Hessian.\n");

	auto N = get_num_basis();
	auto D = get_num_dimensions();

	SGVector<float64_t> xi_hessian_diag(D);
	SGVector<float64_t> beta_sum_hessian_diag(D);

	Map<VectorXd> eigen_xi_hessian_diag(xi_hessian_diag.vector, D);
	Map<VectorXd> eigen_beta_sum_hessian_diag(beta_sum_hessian_diag.vector, D);

	eigen_xi_hessian_diag = VectorXd::Zero(D);
	eigen_beta_sum_hessian_diag = VectorXd::Zero(D);

	Map<VectorXd> eigen_beta(m_beta.vector, get_system_size());

	for (auto a=0; a<N; a++)
	{
		SGVector<float64_t> beta_a(eigen_beta.segment(a*D, D).data(), D, false);

		hessian_diag_xi_add(a, idx_test, xi_hessian_diag);

		for (auto i=0; i<D; i++)
		{
			eigen_beta_sum_hessian_diag[i] += m_kernel->dx_i_dx_j_dx_k_dot_vec_component(
					a, idx_test, beta_a, i, i);
		}
	}

	auto result = beta_sum_hessian_diag;
	hessian_diag_xi_result(xi_hessian_diag, result);

	return result;
}

index_t Full::get_system_size() const
{
	auto N = get_num_basis();
	auto D = get_num_dimensions();

	return N*D;
}

//SGVector<float64_t> Full::leverage() const
//{
//
//	auto leverage = SGVector<float64_t>(ND);
//
//	SG_SINFO("Computing exact leverage scores using SVD.\n");
//
//	auto A = Map<MatrixXd>(build_system().first.matrix, ND+1, ND+1);
//
////	SelfAdjointEigenSolver<MatrixXd> solver(A);
////	auto s = solver.eigenvalues();
////	auto U = solver.eigenvectors();
////
////	switch (solver.info())
////	{
////	case NumericalIssue:
////		SG_SWARNING("Numerical problems computing Eigendecomposition.\n");
////		break;
////	case NoConvergence:
////		SG_SWARNING("No convergence computing Eigendecomposition.\n");
////		break;
////	default:
////		break;
////	}
//
//	// using SVD here since eigen3's self-adjoint eigenvalue produces negatives
//	JacobiSVD<MatrixXd> solver(A.block(1,1,ND,ND), Eigen::ComputeThinU);
//	auto s = solver.singularValues().array().pow(2);
//	auto U = solver.matrixU();
//
//	SG_SINFO("Eigenspectrum range is [%f, %f], or [exp(%f), exp(%f)].\n",
//			s.array().minCoeff(), s.array().maxCoeff(),
//			CMath::log(s.array().minCoeff()), CMath::log(s.array().maxCoeff()));
//
//	for (auto i=0; i<ND; i++)
//	{
//		leverage[i]=0;
//		for (auto j=0; j<ND; j++)
//			leverage[i] += s[j] / (s[j]+ND*m_lambda)*pow(U(i,j), 2);
//	}
//
//	return leverage;
//	return SGVector<float64_t>();
//}
