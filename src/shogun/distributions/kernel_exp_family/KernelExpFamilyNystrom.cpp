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
#include <shogun/io/SGIO.h>
#include <shogun/distributions/kernel_exp_family/KernelExpFamilyNystrom.h>
#include <shogun/distributions/kernel_exp_family/impl/KernelExpFamilyNystromImpl.h>

using namespace shogun;

CKernelExpFamilyNystrom::CKernelExpFamilyNystrom() : CKernelExpFamily()
{
	m_impl=NULL;
}

CKernelExpFamilyNystrom::CKernelExpFamilyNystrom(SGMatrix<float64_t> data,
			float64_t sigma, float64_t lambda, SGVector<index_t> rkhs_basis_inds)
			: CKernelExpFamily()
{
	REQUIRE(data.matrix, "Given observations cannot be empty.\n");
	REQUIRE(data.num_rows>0, "Dimension of given observations (%d) must be positive.\n", data.num_rows);
	REQUIRE(data.num_cols>0, "Number of given observations (%d) must be positive.\n", data.num_cols);
	REQUIRE(sigma>0, "Given sigma (%f) must be positive.\n", sigma);
	REQUIRE(lambda>0, "Given lambda (%f) must be positive.\n", lambda);

	auto m=rkhs_basis_inds.vlen;
	auto N=data.num_cols;
	auto D=data.num_rows;
	auto ND=N*D;
	REQUIRE(m>0, "Given number of RKHS basis functions (%d) must be positive.\n", m);
	REQUIRE(m>0, "Given indices of RKHS basis functions cannot be empty.\n", rkhs_basis_inds.vector);

	for (auto i=0; i<m; i++)
	{
		REQUIRE(rkhs_basis_inds[i]>=0, "RKHS basis index at position %d (%d) must be positive or zero.\n",
				i, rkhs_basis_inds[i]);
		REQUIRE(rkhs_basis_inds[i]<ND, "RKHS basis index at position %d(%d) must be smaller than N*D=%d*$d=%d.\n",
				i, rkhs_basis_inds[i], N, D, ND);

	}

	m_impl = new KernelExpFamilyNystromImpl(data, sigma, lambda, rkhs_basis_inds);
}

CKernelExpFamilyNystrom::CKernelExpFamilyNystrom(SGMatrix<float64_t> data,
			float64_t sigma, float64_t lambda, index_t num_rkhs_basis) : CKernelExpFamily()
{
	m_impl = new KernelExpFamilyNystromImpl(data, sigma, lambda, num_rkhs_basis);
}

CKernelExpFamilyNystrom::~CKernelExpFamilyNystrom()
{
	delete m_impl;
	m_impl=NULL;
}

