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
	return m_kernel->get_num_dimensions();
}

index_t Base::get_num_lhs() const
{
	return m_kernel->get_num_lhs();
}

void Base::set_test_data(SGMatrix<float64_t> X)
{
	m_kernel->set_rhs(X);
	m_kernel->precompute();
}

void Base::set_test_data(SGVector<float64_t> x)
{
	set_test_data(SGMatrix<float64_t>(x));
}

index_t Base::get_num_rhs() const
{
	return m_kernel->get_num_rhs();
}

Base::Base(SGMatrix<float64_t> data,
		kernel::Base* kernel, float64_t lambda)
{
	m_kernel = kernel;
	m_kernel->set_lhs(data);
	m_kernel->set_rhs(data);
	m_lambda = lambda;

	SG_SINFO("Problem size is N=%d, D=%d.\n", get_num_lhs(), get_num_dimensions());
	m_kernel->precompute();
}

Base::~Base()
{
	delete m_kernel;
}

void Base::fit()
{
	SG_SINFO("Building system.\n");
	auto A_b = build_system();

	SG_SINFO("Solving system of size %d.\n", A_b.second.vlen);
	solve_and_store(A_b.first, A_b.second);
}

void Base::solve_and_store(const SGMatrix<float64_t>& A, const SGVector<float64_t>& b)
{
	auto eigen_A = Map<MatrixXd>(A.matrix, A.num_rows, A.num_cols);
	auto eigen_b = Map<VectorXd>(b.vector, b.vlen);

	m_alpha_beta = SGVector<float64_t>(b.vlen);
	auto eigen_alpha_beta = Map<VectorXd>(m_alpha_beta.vector, m_alpha_beta.vlen);

	SG_SINFO("Computing LDLT Cholesky.\n");
	eigen_alpha_beta = eigen_A.ldlt().solve(eigen_b);
}

SGVector<float64_t> Base::log_pdf(const SGMatrix<float64_t> X)
{
	set_test_data(X);
	auto N_test = get_num_rhs();
	SGVector<float64_t> result(N_test);
#pragma omp parallel for
	for (auto i=0; i<N_test; ++i)
		result[i] = log_pdf(i);

	return result;
}

float64_t Base::log_pdf(SGVector<float64_t> x)
{
	set_test_data(x);
	return log_pdf(0);
}

SGVector<float64_t> Base::grad(SGVector<float64_t> x)
{
	set_test_data(x);
	return grad(0);
}

