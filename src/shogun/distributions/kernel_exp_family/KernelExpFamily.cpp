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
#include <shogun/distributions/kernel_exp_family/impl/KernelExpFamilyImpl.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CKernelExpFamily::CKernelExpFamily() : CSGObject()
{
	m_impl=NULL;
}

CKernelExpFamily::CKernelExpFamily(SGMatrix<float64_t> data,
			float64_t sigma, float64_t lambda, float memory_limit_gib) : CSGObject()
{
	REQUIRE(data.matrix, "Given observations cannot be empty\n");
	REQUIRE(data.num_rows>0, "Dimension of given observations (%d) must be positive.\n", data.num_rows);
	REQUIRE(data.num_cols>0, "Number of given observations (%d) must be positive.\n", data.num_cols);
	REQUIRE(sigma>0, "Given sigma (%f) must be positive.\n", sigma);
	REQUIRE(lambda>0, "Given lambda (%f) must be positive.\n", lambda);

	m_impl = new KernelExpFamilyImpl(data, sigma, lambda);

	auto N =  m_impl->get_num_data();
	auto D = m_impl->get_num_dimensions();
	auto ND = D*N;
	auto system_size = (ND+1)*(ND+1) + (ND+1);
	auto memory_required_gib = system_size * sizeof(float64_t) / 8.0 / 1024.0 / 1024.0 / 1024.0;
	if (memory_required_gib > memory_limit_gib)
	{
		SG_ERROR("The problem's size (N=%d, D=%d) will at least use %f GiB of computer memory, "
				"which is above the set limit (%f). "
				"In order to remove this error, increase this limit in the constructor.\n",
				N, D, memory_required_gib, memory_limit_gib);
	}
}

CKernelExpFamily::~CKernelExpFamily()
{
	delete m_impl;
	m_impl=NULL;
}

void CKernelExpFamily::fit()
{
	m_impl->fit();
}

float64_t CKernelExpFamily::log_pdf(SGVector<float64_t> x)
{
	auto D = m_impl->get_num_dimensions();
	REQUIRE(x.vector, "Given data point cannot be empty\n");
	REQUIRE(x.vector, "Dimension of given data point (%d) must match the estimator's (%d)\n", x.vlen, D);

	return m_impl->log_pdf(x);
}

SGVector<float64_t> CKernelExpFamily::grad(SGVector<float64_t> x)
{
	auto D = m_impl->get_num_dimensions();
	REQUIRE(x.vector, "Given data point cannot be empty\n");
	REQUIRE(x.vlen==D, "Dimension of given data point (%d) must match the estimator's (%d)\n", x.vlen, D);

	return m_impl->grad(x);
}

SGVector<float64_t> CKernelExpFamily::log_pdf_multiple(SGMatrix<float64_t> X)
{
	auto D = m_impl->get_num_dimensions();
	REQUIRE(X.matrix, "Given observations cannot be empty\n");
	REQUIRE(X.num_rows==D, "Dimension of given observations (%d) must match the estimator's (%d)\n", X.num_rows, D);
	REQUIRE(X.num_cols>0, "Number of given observations (%d) must be positive.\n", X.num_cols);

	SGVector<float64_t> result(X.num_cols);
#pragma omp for
	for (auto i=0; i<X.num_cols; i++)
	{
		SGVector<float64_t> x(X.get_column_vector(i), m_impl->get_num_dimensions(), false);
		result[i] = m_impl->log_pdf(x);
	}
	return result;
}

SGVector<float64_t> CKernelExpFamily::get_alpha_beta()
{
	return m_impl->get_alpha_beta();
}
