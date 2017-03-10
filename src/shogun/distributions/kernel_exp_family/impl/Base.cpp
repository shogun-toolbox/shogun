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

#include "Base.h"

using namespace shogun;
using namespace shogun::kernel_exp_family_impl;
using namespace Eigen;

index_t Base::get_num_dimensions() const
{
	return m_basis.num_rows;
}

index_t Base::get_num_basis() const
{
	return m_basis.num_cols;
}

void Base::set_data(SGMatrix<float64_t> X)
{
	m_data = X;
	m_kernel->set_rhs(X);
	m_kernel->precompute();
}

void Base::set_data(SGVector<float64_t> x)
{
	set_data(SGMatrix<float64_t>(x));
}

void Base::reset_data()
{
	set_data(m_basis);
}

index_t Base::get_num_data() const
{
	return m_data.num_cols;
}

//const SGVector<float64_t> Base::get_lhs_point(index_t i) const
//{
//	return SGVector<float64_t>(m_lhs.get_column_vector(i), get_num_dimensions(), false);
//}
//
//const SGVector<float64_t> Base::get_rhs_point(index_t i) const
//{
//	return SGVector<float64_t>(m_test_data.get_column_vector(i), get_num_dimensions(), false);
//}

Base::Base(SGMatrix<float64_t> data,
		kernel::Base* kernel, float64_t lambda)
{
	m_kernel = kernel;
	m_lambda = lambda;
	set_basis_and_data(data, data);

	SG_SINFO("Problem size is N=%d, D=%d.\n", get_num_basis(), get_num_dimensions());
}

void Base::set_basis_and_data(SGMatrix<float64_t> basis,
		SGMatrix<float64_t> data)
{
	m_basis = basis;
	m_data = data;

	m_kernel->set_lhs(basis);
	m_kernel->set_rhs(data);
	m_kernel->precompute();
}

Base::~Base()
{
	delete m_kernel;
}

bool Base::is_basis_equal_data() const
{
	// check for same memory location and same dimensions
	if ((m_data.matrix != m_basis.matrix) || (m_data.num_rows != m_basis.num_rows)
			|| (m_data.num_cols != m_basis.num_cols))
		return false;
	else
		return true;
}

float64_t Base::score() const
{
	// TODO check for rounding errors as Python implementation differs for Hessian diagonal
	auto N_test = get_num_data();
	auto D = get_num_dimensions();

	float64_t score = 0.0;

#pragma omp parallel for reduction (+:score)
	for (auto i=0; i<N_test; ++i)
	{
		auto gradient = grad(i);
		auto eigen_gradient = Map<VectorXd>(gradient.vector, D);
		score += 0.5 * eigen_gradient.squaredNorm();

		auto hessian_diag = this->hessian_diag(i);
		auto eigen_hessian_diag = Map<VectorXd>(hessian_diag.vector, D);
		score += eigen_hessian_diag.sum();
	}

	return score / N_test;
}

SGVector<float64_t> Base::log_pdf() const
{
	auto N_test = get_num_data();
	SGVector<float64_t> result(N_test);
#pragma omp parallel for
	for (auto i=0; i<N_test; ++i)
		result[i] = this->log_pdf(i);

	return result;
}

SGMatrix<float64_t> Base::grad() const
{
	auto N_test = get_num_data();
	auto D = get_num_dimensions();

	SGMatrix<float64_t> result(D, N_test);
#pragma omp parallel for
	for (auto i=0; i<N_test; ++i)
	{
		// TODO don't copy this vector around
		auto grad = this->grad(i);
		memcpy(result.get_column_vector(i), grad.vector, D*sizeof(float64_t));
	}

	return result;
}
