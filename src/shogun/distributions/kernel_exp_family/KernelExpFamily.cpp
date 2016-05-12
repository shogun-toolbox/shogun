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
#include <shogun/features/DenseFeatures.h>
#include <shogun/distributions/kernel_exp_family/KernelExpFamily.h>
#include <shogun/distributions/kernel_exp_family/impl/KernelExpFamilyImpl.h>

using namespace shogun;

CKernelExpFamily::CKernelExpFamily() : CSGObject()
{
	m_impl=NULL;
}

CKernelExpFamily::CKernelExpFamily(SGMatrix<float64_t> data,
			float64_t sigma, float64_t lambda) : CSGObject()
{
	REQUIRE(data.matrix, "Data matrix must be set!\n");
	REQUIRE(data.num_rows>0, "Data dimension (%) must be positive\n", data.num_rows);
	REQUIRE(data.num_cols>1, "Number of data (%) must be at least 2\n", data.num_cols);

	m_impl = new KernelExpFamilyImpl(data, sigma, lambda);
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
	return m_impl->log_pdf(x);
}

SGVector<float64_t> CKernelExpFamily::log_pdf_multiple(SGMatrix<float64_t> X)
{
	SGVector<float64_t> result(X.num_rows);
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
