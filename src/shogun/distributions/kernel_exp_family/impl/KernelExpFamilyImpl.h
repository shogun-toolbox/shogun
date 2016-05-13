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
#include <utility>


/* TODO
 *
 * -  The kernel can be optional later on. For now we can just fix it (too many
 *   derivatives to be implemented for now.). We can make it a simple class where
 *   people who want other kernels can overload methods in.
 *   NOTE: Some functions should should take a reference to the return type as
 *         parameter, which allows for pre-allocation of that memory.
 *         Should investigate whether that is needed before doing it. There might
 *         be a few cases where we can avoid using double memory usage peaks for the
 *         big matrices.
 * - There are various TODOs in the code that address the question whether
 *   vectorization makes sense. Check these.
 * - The Gaussian kernel implemented should get doxygen math of what it does,
 *   in particular in the derivatives.
 * - Benchmark and profile the code and investigate in particular whether it has
 *   a lot of cache misses and how we can avoid that. Investigate how it scales
 *   with multiple cores. Investigate how much memory it uses for larger datasets.
 * - Profile memory usage of full vs Nystrom.
 * - Nystrom can be slower than full (N=1000, D=5). Why is that?
 *
 *   Optimizations:
 * - See the various TODOs in the code
 * - Nystrom: Does it make sense to store another (subsetted) copy of the data
 *   to increase speed when looping over it?
 * - How does openmp parallelization affect speed (ask Rahul about data vs job parallel)
 * - Traverse data once and compute the A matrix elements accordingly?
 *   Just like in permutation tests
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

	// new data
	SGVector<float64_t> kernel_dx(const SGVector<float64_t>& a, index_t idx_b);
	SGVector<float64_t> kernel_dx_dx(const SGVector<float64_t>& a, index_t idx_b);
	SGMatrix<float64_t> kernel_dx_i_dx_i_dx_j(const SGVector<float64_t>& a, index_t idx_b);
	SGMatrix<float64_t> kernel_dx_i_dx_j(const SGVector<float64_t>& a, index_t idx_b);

	SGVector<float64_t> compute_h();
	float64_t compute_xi_norm_2();
	std::pair<SGMatrix<float64_t>, SGVector<float64_t>> build_system();
	void fit();
	float64_t log_pdf(const SGVector<float64_t>& x);
	SGVector<float64_t> grad(const SGVector<float64_t>& x);

	SGVector<float64_t> get_alpha_beta() { return m_alpha_beta; }

	index_t get_num_dimensions();
	index_t get_num_data();


protected:
	SGMatrix<float64_t> m_data;
	float64_t m_sigma;
	float64_t m_lambda;

	SGVector<float64_t> m_alpha_beta;
};

}
#endif // KERNEL_EXP_FAMILY_IMPL__
