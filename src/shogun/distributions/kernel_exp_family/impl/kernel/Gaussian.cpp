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

#include "Gaussian.h"

using namespace shogun;
using namespace shogun::kernel_exp_family_impl::kernel;
using namespace Eigen;

Gaussian::Gaussian(float64_t sigma) : Base()
{
	m_sigma = sigma;
}

float64_t Gaussian::kernel(index_t idx_a, index_t idx_b) const
{
	return CMath::exp(-sq_difference_norm(idx_a, idx_b) / m_sigma);
}

float64_t Gaussian::sq_difference_norm(index_t idx_a,  index_t idx_b) const
{
	if (m_sq_difference_norms.matrix)
		return m_sq_difference_norms(idx_a, idx_b);

	SGVector<float64_t> diff = difference(idx_a, idx_b);
	return sq_difference_norm(diff);
}

float64_t Gaussian::sq_difference_norm(const SGVector<float64_t>& diff) const
{
	Map<VectorXd> eigen_diff(diff.vector, diff.vlen);
	return eigen_diff.squaredNorm();
}

SGVector<float64_t> Gaussian::difference(index_t idx_a, index_t idx_b) const
{
	auto D = get_num_dimensions();

	if (m_differences.matrix)
	{
		auto N_rhs = m_rhs.matrix ? get_num_rhs(): get_num_lhs();
		return SGVector<float64_t>(m_differences.get_column_vector(idx_a*N_rhs+idx_b), D, false);
	}

	SGVector<float64_t> result(D);
	difference(idx_a, idx_b, result);
	return result;
}

void Gaussian::difference(index_t idx_a, index_t idx_b,
		SGVector<float64_t>& result) const
{
	auto D = get_num_dimensions();
	if (m_differences.matrix)
	{
		auto N_rhs = m_rhs.matrix ? get_num_rhs() : get_num_lhs();
		memcpy(result.vector,
				m_differences.get_column_vector(idx_a*N_rhs+idx_b),
				sizeof(float64_t)*D);
	}
	else
	{
		Map<VectorXd> x(m_lhs.get_column_vector(idx_a), D);
		float64_t* right_pointer = m_rhs.matrix ?
				m_rhs.get_column_vector(idx_b) : m_lhs.get_column_vector(idx_b);
		Map<VectorXd> y(right_pointer, D);

		Map<VectorXd> eigen_diff(result.vector, D);
		eigen_diff = y-x;
	}
}

SGMatrix<float64_t> Gaussian::dx_i_dx_i_dx_j(index_t idx_a, index_t idx_b) const
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

SGMatrix<float64_t> Gaussian::dx_i_dx_j(index_t idx_a, index_t idx_b) const
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

SGMatrix<float64_t> Gaussian::dx_dx_dy(index_t idx_a, index_t idx_b) const
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

float64_t Gaussian::dx_dx_dy_dy_sum(index_t idx_a, index_t idx_b) const
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

SGMatrix<float64_t> Gaussian::dx_dx_dy_dy(index_t idx_a, index_t idx_b) const
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

SGMatrix<float64_t> Gaussian::dx_dy(index_t idx_a, index_t idx_b) const
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

SGVector<float64_t> Gaussian::dx_dx(index_t idx_a, index_t idx_b) const
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

SGMatrix<float64_t> Gaussian::dx_dy_all() const
{
	auto D = get_num_dimensions();
	auto N = get_num_lhs();
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
			SGMatrix<float64_t> h=dx_dy(idx_a, idx_b);
			eigen_result.block(r_start, c_start, D, D) = Map<MatrixXd>(h.matrix, D, D);
			eigen_result.block(c_start, r_start, D, D) = eigen_result.block(r_start, c_start, D, D);
		}

	return result;
}

SGVector<float64_t> Gaussian::dx(index_t idx_a, index_t idx_b) const
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


float64_t Gaussian::difference_component(index_t idx_a, index_t idx_b, index_t i) const
{
	auto D = get_num_dimensions();

	if (m_differences.matrix)
	{
		auto N_rhs = m_rhs.matrix ? get_num_rhs(): get_num_lhs();
		auto diff = SGVector<float64_t>(m_differences.get_column_vector(idx_a*N_rhs+idx_b), D, false);
		return diff[i];
	}

	Map<VectorXd> x(m_lhs.get_column_vector(idx_a), D);
	float64_t* right_pointer = m_rhs.matrix ?
			m_rhs.get_column_vector(idx_b) : m_lhs.get_column_vector(idx_b);
	Map<VectorXd> y(right_pointer, D);

	return y[i]-x[i];
}

float64_t Gaussian::dx_dy_component(const index_t idx_a, const index_t idx_b, const index_t i, const index_t j) const
{
	auto D = get_num_dimensions();

	Map<VectorXd> x(m_lhs.get_column_vector(idx_a), D);
	Map<VectorXd> y(m_lhs.get_column_vector(idx_b), D);

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

float64_t Gaussian::dx_dx_dy_dy_component(index_t idx_a, index_t idx_b, index_t i, index_t j) const
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

float64_t Gaussian::dx_dx_dy_component(index_t idx_a, index_t idx_b, index_t i, index_t j) const
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

float64_t Gaussian::dx_component(index_t idx_a, index_t idx_b, index_t i) const
{
	//k = gaussian_kernel(x_2d, y_2d, sigma)
	auto diff_i = difference_component(idx_a, idx_b, i);
	auto k=kernel(idx_a, idx_b);

	return 2*k*diff_i/m_sigma;
}

float64_t Gaussian::dx_dx_component(index_t idx_a, index_t idx_b, index_t i) const
{
	auto diff_i = difference_component(idx_a, idx_b, i);
	auto k=kernel(idx_a, idx_b);

	return k*(pow(diff_i,2)*pow(2.0/m_sigma, 2) -2.0/m_sigma);
}

SGVector<float64_t> Gaussian::dx_i_dx_i_dx_j_component(index_t idx_a, index_t idx_b, index_t i) const
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

SGVector<float64_t> Gaussian::dx_i_dx_j_component(index_t idx_a, index_t idx_b, index_t i) const
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

void Gaussian::precompute()
{
	// remove potentially previously precomputed quantities to make calls below
	// not use existing matrices
	m_sq_difference_norms = SGMatrix<float64_t>();
	m_differences = SGMatrix<float64_t>();

	SGMatrix<float64_t> sq_difference_norms;
	SGMatrix<float64_t> differences;

	auto D = get_num_dimensions();

	// distinguish symmetric and non-symmetric case
	if (!m_rhs.matrix)
	{
		auto N = get_num_lhs();
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
		auto N_lhs = get_num_lhs();
		auto N_rhs = get_num_rhs();
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
