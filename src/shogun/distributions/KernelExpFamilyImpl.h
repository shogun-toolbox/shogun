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

#ifndef KERNEL_EXP_FAMILY_IMPL__
#define KERNEL_EXP_FAMILY_IMPL__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/kernel/Kernel.h>
#include <utility>


namespace shogun
{

class KernelExpFamilyImpl
{
public :
	KernelExpFamilyImpl(SGMatrix<float64_t> data, float64_t sigma, float64_t lambda);

	// for training
	float64_t kernel(index_t idx_a, const index_t idx_b);
	SGMatrix<float64_t> kernel_dx_dx_dy(index_t idx_a, index_t idx_b);
	SGMatrix<float64_t> kernel_dx_dx_dy_dy(index_t idx_a, index_t idx_b);
	SGMatrix<float64_t> kernel_hessian(index_t idx_a, index_t idx_b);
	SGMatrix<float64_t> kernel_hessian_all();
	float64_t kernel_hessian_i_j(index_t idx_a, index_t idx_b, index_t i, index_t j);

	// for new data
	SGVector<float64_t> kernel_dx(const SGVector<float64_t>& a, index_t idx_b);
	SGVector<float64_t> kernel_dx_dx(const SGVector<float64_t>& a, index_t idx_b);

	// full estimator
	SGVector<float64_t> compute_h();
	float64_t compute_xi_norm_2();
	std::pair<SGMatrix<float64_t>, SGVector<float64_t>> build_system();
	SGVector<float64_t> fit();

	// nystrom approximation
	std::pair<index_t, index_t> idx_to_ai(index_t idx);
	float64_t compute_lower_right_submatrix_element(index_t row_idx, index_t col_idx);
	SGVector<float64_t> compute_first_row_no_storing();
	std::pair<SGMatrix<float64_t>, SGVector<float64_t>> build_system_nystrom(SGVector<index_t> inds);
	SGVector<float64_t> fit_nystrom(SGVector<index_t> inds);
	SGMatrix<float64_t> pinv(SGMatrix<float64_t> A);

	// model
	float64_t log_pdf(const SGVector<float64_t>& x, const SGVector<float64_t>& alpha_beta);
	SGVector<float64_t> grad(SGVector<float64_t> x);

protected:
	index_t get_dimension();
	index_t get_num_data();


protected:
	SGMatrix<float64_t> m_data;
	float64_t m_sigma;
	float64_t m_lambda;
};

}
#endif // KERNEL_EXP_FAMILY_IMPL__
