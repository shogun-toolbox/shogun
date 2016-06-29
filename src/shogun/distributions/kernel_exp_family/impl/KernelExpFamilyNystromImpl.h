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

#ifndef KERNEL_EXP_FAMILY_NYSTROM_IMPL__
#define KERNEL_EXP_FAMILY_NYSTROM_IMPL__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include "KernelExpFamilyImpl.h"


namespace shogun
{

class KernelExpFamilyNystromImpl : public KernelExpFamilyImpl
{
public :
	KernelExpFamilyNystromImpl(SGMatrix<float64_t> data, float64_t sigma, float64_t lambda,
				index_t num_rkhs_basis);
	KernelExpFamilyNystromImpl(SGMatrix<float64_t> data, float64_t sigma, float64_t lambda,
			SGVector<index_t> rkhs_basis_inds);

	virtual ~KernelExpFamilyNystromImpl() {};

	void sub_sample_rkhs_basis(index_t num_rkhs_basis);

	// overloaded
	float64_t compute_xi_norm_2() const;
	SGVector<float64_t> compute_h() const;

	float64_t difference_component(index_t idx_a, index_t idx_b, index_t i) const;
	float64_t kernel_hessian_component(const index_t idx_a, index_t idx_b, index_t i, index_t j) const;
	float64_t kernel_dx_dx_dy_dy_component(index_t idx_a, index_t idx_b, index_t i, index_t j) const;
	float64_t kernel_dx_dx_dy_component(index_t idx_a, index_t idx_b, index_t i, index_t j) const;
	float64_t kernel_dx_component(index_t idx_a, index_t idx_b, index_t i) const;
	float64_t kernel_dx_dx_component(index_t idx_a, index_t idx_b, index_t i) const;
	SGVector<float64_t> kernel_dx_i_dx_i_dx_j_component(index_t idx_a, index_t idx_b, index_t i) const;
	SGVector<float64_t> kernel_dx_i_dx_j_component(index_t idx_a, index_t idx_b, index_t i) const;

	virtual std::pair<SGMatrix<float64_t>, SGVector<float64_t>> build_system() const;

	std::pair<index_t, index_t> idx_to_ai(index_t idx) const;

	virtual float64_t log_pdf(index_t idx_test) const;
	virtual SGVector<float64_t> grad(index_t idx_test) const;

	using KernelExpFamilyImpl::solve_and_store;
	using KernelExpFamilyImpl::log_pdf;
	using KernelExpFamilyImpl::grad;

	static SGMatrix<float64_t> pinv_self_adjoint(const SGMatrix<float64_t>& A);

protected:
	index_t get_num_rkhs_basis() const;

protected:
	SGVector<index_t> m_rkhs_basis_inds;
};

}
#endif // KERNEL_EXP_FAMILY_NYSTROM_IMPL__
