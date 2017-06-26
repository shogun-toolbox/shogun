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

#ifndef KERNEL_EXP_FAMILY_IMPL_NYSTROM__
#define KERNEL_EXP_FAMILY_IMPL_NYSTROM__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include <memory>

#include "Base.h"
#include "Full.h"

namespace shogun
{

namespace kernel_exp_family_impl
{
class Nystrom : public Full
{
public :
	Nystrom(SGMatrix<float64_t> data, SGMatrix<float64_t> basis,
			std::shared_ptr<kernel::Base> kernel, float64_t lambda, float64_t lambda_l2=0.0);

	Nystrom(SGMatrix<float64_t> data, SGVector<index_t> basis_inds,
			std::shared_ptr<kernel::Base> kernel, float64_t lambda, float64_t lambda_l2=0.0);

	Nystrom(SGMatrix<float64_t> data, index_t num_subsample_basis,
			std::shared_ptr<kernel::Base> kernel, float64_t lambda, float64_t lambda_l2=0.0);

	virtual ~Nystrom() {};

	virtual void fit();

	// TODO these should go to the lib
	static SGMatrix<float64_t> pinv_self_adjoint(const SGMatrix<float64_t>& A);
	static SGVector<index_t> choose_m_in_n(index_t m, index_t n, bool sorted=true);
	static SGMatrix<float64_t> subsample_matrix_cols(const SGVector<index_t>& col_inds,
			const SGMatrix<float64_t>& mat);

	// modularisation of the fit method
	virtual bool basis_is_subsampled_data() const { return m_basis_inds.vlen; }
	virtual index_t get_system_size() const;
	SGMatrix<float64_t> compute_system_matrix();
	SGVector<float64_t> compute_system_vector() const;


	virtual SGMatrix<float64_t> subsample_G_mm_from_G_mn(const SGMatrix<float64_t>& G_mn) const;
	virtual SGMatrix<float64_t> compute_G_mn() const;
	virtual SGMatrix<float64_t> compute_G_mm(); // TODO this should be const!
	SGVector<float64_t> solve_system(const SGMatrix<float64_t>& system_matrix,
			const SGVector<float64_t>& system_vector) const;

	// overloading base class methods as no-ops to get rid of xi parts
	virtual void log_pdf_xi_add(index_t basis_ind, index_t idx_test, float64_t& xi) const {};
	virtual void log_pdf_xi_result(float64_t xi, float64_t& result) const {};
	virtual void grad_xi_add(index_t basis_ind, index_t idx_test,
			SGVector<float64_t>& xi_grad) const {};
	virtual void grad_xi_result(const SGVector<float64_t>& xi,
			 SGVector<float64_t>& result) const {};
	virtual void hessian_xi_add(index_t basis_ind, index_t idx_test,
			SGMatrix<float64_t>& xi_hessian) const {};
	virtual void hessian_xi_result(const SGMatrix<float64_t>& xi_hessian,
			 SGMatrix<float64_t>& result) const {};
	virtual void hessian_diag_xi_add(index_t basis_ind, index_t idx_test,
			SGVector<float64_t>& xi_hessian_diag) const {};
	virtual void hessian_diag_xi_result(const SGVector<float64_t>& xi_hessian_diag,
			SGVector<float64_t>& result) const {};
protected:

	float64_t m_lambda_l2;
	SGVector<index_t> m_basis_inds;
};
};

}
#endif // KERNEL_EXP_FAMILY_IMPL_NYSTROM__
