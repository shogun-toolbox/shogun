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
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <memory>

#include "kernel/Base.h"
#include "Sum.h"
#include "Nystrom.h"
#include "Full.h"

#include <shogun/io/SGIO.h>  // XXX

using namespace shogun;
using namespace shogun::kernel_exp_family_impl;
using namespace Eigen;

Sum::Sum(SGMatrix<float64_t> data, SGMatrix<float64_t> basis,
		           std::shared_ptr<kernel::Base> kernel, float64_t lambda,
           		   float64_t lambda_l2, bool init_base_and_data)
	: Nystrom(data, basis, kernel, lambda, lambda_l2, init_base_and_data)
{ }

Sum::Sum(SGMatrix<float64_t> data, SGVector<index_t> basis_inds,
		           std::shared_ptr<kernel::Base> kernel, float64_t lambda,
			       float64_t lambda_l2, bool init_base_and_data)
    : Nystrom(data, basis_inds, kernel, lambda, lambda_l2, init_base_and_data)
{ }

Sum::Sum(SGMatrix<float64_t> data, index_t num_subsample_basis,
	               std::shared_ptr<kernel::Base> kernel, float64_t lambda,
			       float64_t lambda_l2, bool init_base_and_data)
	: Nystrom(data, num_subsample_basis, kernel, lambda, lambda_l2, init_base_and_data)
{ }

index_t Sum::get_system_size() const
{
	return get_num_basis();
}

SGMatrix<float64_t> Sum::subsample_G_mm_from_G_mn(const SGMatrix<float64_t>& G_mn) const
{
	SG_SERROR("Can't subsample G_mm from G_mn for Sum");

	SGMatrix<float64_t> return_something(0, 0);
	return return_something;
}
bool Sum::can_subsample_G_mm_from_G_mn() const
{
	return false;
}


SGMatrix<float64_t> Sum::compute_G_mn() const
{
	auto G_mn = m_kernel->dx_sum_dy_all();
	return G_mn;
}

SGMatrix<float64_t> Sum::compute_G_mm() const
{
	auto basis = m_basis;
	auto data = m_data;

	auto kernel=m_kernel->shallow_copy();
	kernel->set_lhs(basis);
	kernel->set_rhs();  // make symmetric
	kernel->precompute();

	auto G_mm = kernel->sum_dx_sum_dy_all();
	return G_mm;
}

SGVector<float64_t> Sum::compute_h() const
{
	auto N_basis = get_num_basis();
	auto N_data = get_num_data();

	SGVector<float64_t> h(N_basis);
	Map<VectorXd> eigen_h(h.vector, N_basis);
	eigen_h = VectorXd::Zero(N_basis);

#pragma omp parallel for
	for (auto idx_a = 0; idx_a < N_basis; idx_a++)
		for (auto idx_b = 0; idx_b < N_data; idx_b++)
			eigen_h(idx_a) += m_kernel->sum_dx_sum_dy_dy(idx_a, idx_b);

	eigen_h /= N_data;
	return h;
}


float64_t Sum::log_pdf(index_t idx_test) const
{
	auto N_basis = get_num_basis();

	float64_t beta_sum = 0;
	for (auto idx = 0; idx < N_basis; ++idx)
	{
		beta_sum += m_beta[idx] * m_kernel->sum_dx(idx, idx_test);
	}
	return beta_sum;
}


// TODO: implement these...
SGVector<float64_t> Sum::grad(index_t idx_test) const
{
	SG_SERROR("Sum::grad not implemented yet");
	return SGVector<float64_t>();
}

SGMatrix<float64_t> Sum::hessian(index_t idx_test) const
{
	SG_SERROR("Sum::hessian not implemented yet");
	return SGMatrix<float64_t>();
}

SGVector<float64_t> Sum::hessian_diag(index_t idx_test) const
{
	SG_SERROR("Sum::hessian_diag not implemented yet");
	return SGVector<float64_t>();
}
