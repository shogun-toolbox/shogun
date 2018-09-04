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
#include <shogun/distributions/kernel_exp_family/KernelExpFamily.h>
#include <shogun/io/SGIO.h>
#include <memory>

#include "impl/Full.h"
#include "impl/kernel/Gaussian.h"

using namespace shogun;

CKernelExpFamily::CKernelExpFamily() : CSGObject()
{
	m_impl=NULL;
}

CKernelExpFamily::CKernelExpFamily(SGMatrix<float64_t> data,
			float64_t sigma, float64_t lambda, float64_t base_measure_cov_ridge) : CSGObject()
{
	REQUIRE(data.matrix, "Given observations cannot be empty.\n");
	REQUIRE(data.num_rows>0, "Dimension of given observations (%d) must be positive.\n", data.num_rows);
	REQUIRE(data.num_cols>0, "Number of given observations (%d) must be positive.\n", data.num_cols);
	REQUIRE(sigma>0, "Given sigma (%f) must be positive.\n", sigma);
	REQUIRE(lambda>0, "Given lambda (%f) must be positive.\n", lambda);
	REQUIRE(base_measure_cov_ridge>=0, "Given base measure covariance ridge (%f) must be 0 or positive.\n", base_measure_cov_ridge);

	auto kernel = std::make_shared<kernel_exp_family_impl::kernel::Gaussian>(sigma);
	m_impl = new kernel_exp_family_impl::Full(data, kernel, lambda, base_measure_cov_ridge);
}

CKernelExpFamily::~CKernelExpFamily()
{
	delete m_impl;
	m_impl=NULL;
}

void CKernelExpFamily::fit()
{
	REQUIRE(m_impl->is_basis_equal_data(),
			"Cannot proceed with set data. Reset data!\n");

	m_impl->fit();
}

float64_t CKernelExpFamily::log_pdf(index_t i)
{
	auto N = m_impl->get_num_data();
	REQUIRE(i>=0 && i<N, "Given test data index (%d) must be in [0, %d].\n", i, N-1);
	return m_impl->log_pdf(i);
}

SGVector<float64_t> CKernelExpFamily::log_pdf_multiple()
{
	return m_impl->log_pdf();
}

SGMatrix<float64_t> CKernelExpFamily::grad_multiple()
{
	return m_impl->grad();
}

SGVector<float64_t> CKernelExpFamily::grad(index_t i)
{
	auto N = m_impl->get_num_data();
	REQUIRE(i>=0 && i<N, "Given test data index (%d) must be in [0, %d].\n", i, N-1);
	return m_impl->grad(i);
}

SGMatrix<float64_t> CKernelExpFamily::hessian(index_t i)
{
	auto N = m_impl->get_num_data();
	REQUIRE(i>=0 && i<N, "Given test data index (%d) must be in [0, %d].\n", i, N-1);
	return m_impl->hessian(i);
}

SGVector<float64_t> CKernelExpFamily::hessian_diag(index_t i)
{
	auto N = m_impl->get_num_data();
	REQUIRE(i>=0 && i<N, "Given test data index (%d) must be in [0, %d].\n", i, N-1);
	return m_impl->hessian_diag(i);
}

float64_t CKernelExpFamily::score()
{
	return m_impl->score();
}

//SGVector<float64_t> CKernelExpFamily::leverage()
//{
//	REQUIRE(m_impl->is_test_equals_train_data(),
//			"Cannot proceed with test data. Reset test data!\n");
//	return m_impl->leverage();
//	return SGVector<float64_t>();
//}

SGMatrix<float64_t> CKernelExpFamily::get_matrix(const char* name)
{
	REQUIRE(false, "No matrix with given name (%s).\n", name);

	return SGMatrix<float64_t>();
}

SGVector<float64_t> CKernelExpFamily::get_vector(const char* name)
{
	if (!strcmp(name, "beta"))
		return m_impl->get_beta();
	else
		REQUIRE(false, "No vector with given name (%s).\n", name);

	return SGVector<float64_t>();
}

void CKernelExpFamily::set_vector(const char* name, SGVector<float64_t> vec)
{
	if (!strcmp(name, "beta"))
		m_impl->set_beta(vec);
	else
		REQUIRE(false, "No vector with given name (%s).\n", name);
}

void CKernelExpFamily::reset_data()
{
	m_impl->reset_data();
}

void CKernelExpFamily::set_data(SGMatrix<float64_t> X)
{
	auto D = m_impl->get_num_dimensions();
	REQUIRE(X.matrix, "Given observations cannot be empty.\n");
	REQUIRE(X.num_rows==D, "Dimension of given observations (%d) must match the estimator's (%d).\n", X.num_rows, D);
	REQUIRE(X.num_cols>0, "Number of given observations (%d) must be positive.\n", X.num_cols);
	m_impl->set_data(X);
}

void CKernelExpFamily::set_data(SGVector<float64_t> x)
{
	auto D = m_impl->get_num_dimensions();
	REQUIRE(x.vector, "Given observations cannot be empty.\n");
	REQUIRE(x.vlen==D, "Dimension of given point (%d) must match the estimator's (%d).\n", x.vlen, D);
	m_impl->set_data(x);
}
