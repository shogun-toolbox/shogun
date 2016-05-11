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


/* TODO
 *
 * - Make all methods static functions using raw pointers, N, and D in the arguments
 *   which allows to extract the code from shogun and use it in isolation.
 *   The kernel can be optional later on. For now we can just fix it (too many
 *   derivatives to be implemented for now.). We can make it a simple class where
 *   people who want other kernels can overload methods in.
 *   NOTE: The functions should should take a reference to the return type as
 *         parameter, which allows for pre-allocation of that memory.
 *         Should investigate whether that is needed before doing it. There might
 *         be a few cases where we can avoid using double memory usage peaks for the
 *         big matrices.
 * - In the OOP code, the fit() method should store the alpha, beta inside rather
 *   than returning it.
 * - Make Nystrom a sub-class of the original estimator.
 * - There are various TODOs in the code that address the question whether
 *   vectorization makes sense. Check these.
 * - The Gaussian kernel implemented should get doxygen math of what it does,
 *   in particular in the derivatives.
 * - Benchmark and profile the code and investigate in particular whether it has
 *   a lot of cache misses and how we can avoid that. Investigate how it scales
 *   with multiple cores. Investigate how much memory it uses for larger datasets.
 *
 */

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
	float64_t kernel_dx_i(const SGVector<float64_t>& a, index_t idx_b, index_t i);
	SGVector<float64_t> kernel_dx_dx(const SGVector<float64_t>& a, index_t idx_b);
	float64_t kernel_dx_dx_i(const SGVector<float64_t>& a, index_t idx_b, index_t i);


	// full estimator
	SGVector<float64_t> compute_h();
	float64_t compute_xi_norm_2();
	std::pair<SGMatrix<float64_t>, SGVector<float64_t>> build_system();
	SGVector<float64_t> fit();
	float64_t log_pdf(const SGVector<float64_t>& x, const SGVector<float64_t>& alpha_beta);

	// nystrom approximation
	std::pair<index_t, index_t> idx_to_ai(index_t idx);
	float64_t compute_lower_right_submatrix_element(index_t row_idx, index_t col_idx);
	SGVector<float64_t> compute_first_row_no_storing();
	std::pair<SGMatrix<float64_t>, SGVector<float64_t>> build_system_nystrom(SGVector<index_t> inds);
	SGVector<float64_t> fit_nystrom(SGVector<index_t> inds);
	SGMatrix<float64_t> pinv(SGMatrix<float64_t> A);
	float64_t  log_pdf_nystrom(const SGVector<float64_t>& x, const SGVector<float64_t>& alpha_beta, const SGVector<index_t>& inds);


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
