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
			SGVector<index_t> inds);

	// for training
	float64_t kernel_hessian_component(index_t idx_a, index_t idx_b, index_t i, index_t j);

	// for new data
	float64_t kernel_dx_component(const SGVector<float64_t>& a, index_t idx_b, index_t i);
	float64_t kernel_dx_dx_component(const SGVector<float64_t>& a, index_t idx_b, index_t i);
	SGVector<float64_t> kernel_dx_i_dx_i_dx_j_component(const SGVector<float64_t>& a, index_t idx_b, index_t i);
	SGVector<float64_t> kernel_dx_i_dx_j_component(const SGVector<float64_t>& a, index_t idx_b, index_t i);

	float64_t compute_lower_right_submatrix_element(index_t row_idx, index_t col_idx);
	SGVector<float64_t> compute_first_row_no_storing();
	std::pair<SGMatrix<float64_t>, SGVector<float64_t>> build_system();
	void fit();
	float64_t  log_pdf(const SGVector<float64_t>& x);
	SGVector<float64_t> grad(const SGVector<float64_t>& x);

	std::pair<index_t, index_t> idx_to_ai(const index_t& idx);
	static SGMatrix<float64_t> pinv(const SGMatrix<float64_t>& A);

	SGVector<index_t> get_inds() { return m_inds; }


protected:
	index_t get_num_rkhs_basis();

protected:
	SGVector<index_t> m_inds;
};

}
#endif // KERNEL_EXP_FAMILY_NYSTROM_IMPL__
