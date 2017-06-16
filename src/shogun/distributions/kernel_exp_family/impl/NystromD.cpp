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

#include "kernel/Base.h"
#include "NystromD.h"

#include <vector>

using namespace shogun;
using namespace shogun::kernel_exp_family_impl;
using namespace Eigen;

NystromD::NystromD(SGMatrix<float64_t> data, SGMatrix<bool> basis_mask,
		kernel::Base* kernel, float64_t lambda,
		float64_t lambda_l2) : Nystrom(data, data, kernel, lambda, lambda_l2)
{
	set_basis_inds_from_mask(basis_mask);
}

NystromD::NystromD(SGMatrix<float64_t> data, SGMatrix<float64_t> basis,
		SGMatrix<bool> basis_mask,
		kernel::Base* kernel, float64_t lambda,
		float64_t lambda_l2) : Nystrom(data, basis, kernel, lambda, lambda_l2)
{
	set_basis_inds_from_mask(basis_mask);
}

void NystromD::set_basis_inds_from_mask(const SGMatrix<bool>& basis_mask)
{
	std::vector<index_t> basis_inds;
	int64_t num_mask_elements = (int64_t)basis_mask.num_rows*(int64_t)basis_mask.num_cols;
	for (auto i=0; i<num_mask_elements; i++)
	{
		if (basis_mask.matrix[i])
			basis_inds.push_back(i);
	}

	m_basis_inds = SGVector<index_t>(basis_inds.size());
	memcpy(m_basis_inds.vector, basis_inds.data(), basis_inds.size() * sizeof(index_t));

	SG_SINFO("Using subsampled basis components, %d of %dx%d=%d possible components.\n",
			basis_inds.size(), basis_mask.num_rows, basis_mask.num_cols,
			basis_mask.size());
}

index_t NystromD::get_system_size() const
{
	return m_basis_inds.vlen;
}

SGVector<float64_t> NystromD::compute_h() const
{
	SG_SWARNING("TODO: dont compute all and then sub-sample.\n");
	auto h_full = Nystrom::compute_h();

	SGVector<float64_t> h(get_num_basis());

	// subsample vector entries
	for (auto i=0; i<get_num_basis(); i++)
		h[i] = h_full[m_basis_inds[i]];

	return h;
}

SGMatrix<float64_t> NystromD::compute_G_mn() const
{
	SG_SWARNING("TODO: dont compute all and then sub-sample.\n");
	auto G_mn_full = Nystrom::compute_G_mn();

	auto D = get_num_dimensions();
	auto system_size = get_system_size();
	auto N = get_num_data();
	auto ND = N*D;

	SGMatrix<float64_t> G_mn(system_size, ND);

	// subsample matrix rows
	for (auto i=0; i<ND; i++)
	{
		for (auto j=0; j<system_size; j++)
			G_mn(j,i) = G_mn_full(m_basis_inds[j], i);
	}

	return G_mn;
}

SGMatrix<float64_t> NystromD::compute_G_mm()
{
	SG_SWARNING("TODO: dont compute all and then sub-sample.\n");
	auto G_mm_full = Nystrom::compute_G_mm();
	auto system_size = get_system_size();

	SGMatrix<float64_t> G_mm(system_size, system_size);

	// subsample matrix rows and columns
	for (auto i=0; i<G_mm_full.num_cols; i++)
	{
		for (auto j=0; j<G_mm_full.num_rows; j++)
			G_mm(j,i) = G_mm_full(m_basis_inds[j], m_basis_inds[i]);
	}

	return G_mm;
}

bool NystromD::basis_is_subsampled_data() const
{
	SG_SWARNING("TODO: Optimize case where basis is equal to data!\n");
	return false;
	return m_data == m_basis;
}

SGMatrix<float64_t> NystromD::subsample_G_mm_from_G_mn(const SGMatrix<float64_t>& G_mn) const
{
	//TODO
	ASSERT(false);
	return SGMatrix<float64_t>();
}


