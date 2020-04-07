/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
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
 *
 */

#ifndef GAUSSIANARDSPARSEKERNEL_H
#define GAUSSIANARDSPARSEKERNEL_H

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/GaussianARDKernel.h>

namespace shogun
{

/** @brief Gaussian Kernel with Automatic Relevance Detection with supporting
 * Sparse inference
 *
 * This kernel supports to compute the gradient wrt latent features (inducing points),
 * which are not hyper-parameters of the kernel.
 *
 * */
class GaussianARDSparseKernel: public GaussianARDKernel
{
public:
	/** default constructor */
	GaussianARDSparseKernel();

	/** return what type of kernel we are
	 *
	 * @return kernel type GAUSSIANARD
	 */
	EKernelType get_kernel_type() override { return K_GAUSSIANARDSPARSE; }

	/** return the kernel's name
	 *
	 * @return name GaussianARDSparseKernel
	 */
	const char* get_name() const override { return "GaussianARDSparseKernel"; }

	/** destructor */
	~GaussianARDSparseKernel() override;

private:
	void initialize_sparse_kernel();

public:
	/** constructor
	 *
	 * @param size cache size
	 */
	GaussianARDSparseKernel(int32_t size);

	/** constructor
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @param size cache size
	 */
	GaussianARDSparseKernel(std::shared_ptr<DotFeatures> l, std::shared_ptr<DotFeatures> r,
		int32_t size=10);

	/** @param kernel is casted to GaussianARDSparseKernel, error if not possible
	 * is SG_REF'ed
	 * @return casted GaussianARDSparseKernel object
	 */
	static std::shared_ptr<GaussianARDSparseKernel> obtain_from_generic(const std::shared_ptr<Kernel>& kernel);

	/** return derivative with respect to specified parameter
	 *
	 * @param param the parameter
	 * @param index the index of the element if parameter is a vector or matrix
	 * if the parameter is a matrix, index is the linearized index
	 * of the matrix (column-major)
	 *
	 * @return gradient with respect to parameter
	 */
	SGMatrix<float64_t> get_parameter_gradient(Parameters::const_reference param,
		index_t index=-1) override;

	/** return diagonal part of derivative with respect to specified parameter
	 *
	 * @param param the parameter
	 * @param index the index of the element if parameter is a vector
	 *
	 * @return diagonal part of gradient with respect to parameter
	 */
	SGVector<float64_t> get_parameter_gradient_diagonal(
		Parameters::const_reference param, index_t index=-1) override;
};
}

#endif /* GAUSSIANARDSPARSEKERNEL_H */
