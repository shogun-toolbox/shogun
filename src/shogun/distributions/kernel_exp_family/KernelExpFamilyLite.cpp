/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2018 Dougal Sutherland
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
#include <shogun/io/SGIO.h>
#include <shogun/distributions/kernel_exp_family/KernelExpFamilyLite.h>
#include <shogun/distributions/kernel_exp_family/impl/Lite.h>
#include <shogun/distributions/kernel_exp_family/impl/NystromD.h>
#include <memory>

#include "impl/kernel/Gaussian.h"


using namespace shogun;

CKernelExpFamilyLite::CKernelExpFamilyLite() : CKernelExpFamily()
{
	m_impl=NULL;
}

CKernelExpFamilyLite::CKernelExpFamilyLite(SGMatrix<float64_t> data,
			SGMatrix<float64_t> basis,
			float64_t sigma, float64_t lambda, float64_t lambda_l2)
			: CKernelExpFamily()
{
	REQUIRE(data.matrix, "Given observations cannot be empty.\n");
	REQUIRE(data.num_rows>0, "Dimension of given observations (%d) must be positive.\n", data.num_rows);
	REQUIRE(data.num_cols>0, "Number of given observations (%d) must be positive.\n", data.num_cols);
	REQUIRE(basis.matrix, "Given basis cannot be empty.\n");
	REQUIRE(basis.num_rows>0, "Dimension of given basis (%d) must be positive.\n", basis.num_rows);
	REQUIRE(basis.num_cols>0, "Number of given basis (%d) must be positive.\n", basis.num_cols);
	REQUIRE(sigma>0, "Given sigma (%f) must be positive.\n", sigma);
	REQUIRE(lambda>0, "Given lambda (%f) must be positive.\n", lambda);
	REQUIRE(lambda>=0, "Given L2 lambda (%f) must be >=0.\n", lambda_l2);

	auto kernel = std::make_shared<kernel_exp_family_impl::kernel::Gaussian>(sigma);
	m_impl = new kernel_exp_family_impl::Lite(data, basis, kernel, lambda, lambda_l2);
}

CKernelExpFamilyLite::CKernelExpFamilyLite(SGMatrix<float64_t> data, index_t num_subsample_basis,
				float64_t sigma, float64_t lambda, float64_t lambda_l2)
{
	REQUIRE(data.matrix, "Given observations cannot be empty.\n");
	REQUIRE(data.num_rows>0, "Dimension of given observations (%d) must be positive.\n", data.num_rows);
	REQUIRE(data.num_cols>0, "Number of given observations (%d) must be positive.\n", data.num_cols);
	REQUIRE(num_subsample_basis>0, "Number of subsampled basis (%d) must be positive.\n", num_subsample_basis);
	REQUIRE(num_subsample_basis<=data.num_cols, "Number of subsampled basis (%d) must not exceed number of data (%d).\n",
			num_subsample_basis, data.num_cols);
	REQUIRE(sigma>0, "Given sigma (%f) must be positive.\n", sigma);
	REQUIRE(lambda>0, "Given lambda (%f) must be positive.\n", lambda);
	REQUIRE(lambda>=0, "Given L2 lambda (%f) must be >=0.\n", lambda_l2);

	auto kernel = std::make_shared<kernel_exp_family_impl::kernel::Gaussian>(sigma);
	m_impl = new kernel_exp_family_impl::Lite(data, num_subsample_basis,
													kernel, lambda, lambda_l2);
}

CKernelExpFamilyLite::~CKernelExpFamilyLite()
{
	delete m_impl;
	m_impl=NULL;
}

void CKernelExpFamilyLite::fit()
{
	m_impl->fit();
}

SGVector<index_t> CKernelExpFamilyLite::get_basis_inds() const
{
	return static_cast<kernel_exp_family_impl::Lite *>(m_impl)->get_basis_inds();
}

SGMatrix<float64_t> CKernelExpFamilyLite::get_matrix(const char* name)
{
	if (!strcmp(name, "basis"))
		return static_cast<kernel_exp_family_impl::Lite *>(m_impl)->get_basis();
	else
		return CKernelExpFamily::get_matrix(name);
}
