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

#ifndef KERNEL_EXP_FAMILY_KERNEL_GAUSSIAN__
#define KERNEL_EXP_FAMILY_KERNEL_GAUSSIAN__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include "Base.h"

namespace shogun
{
namespace kernel_exp_family_impl
{
namespace kernel
{

class Gaussian : public Base
{
public :
	Gaussian(float64_t sigma);
	virtual ~Gaussian() {};

	// translation invariant part
	float64_t sq_difference_norm(index_t idx_a, index_t idx_b) const;
	float64_t sq_difference_norm(const SGVector<float64_t>& diff) const;
	SGVector<float64_t> difference(index_t idx_a, index_t idx_b) const;
	void difference(index_t idx_a, index_t idx_b, SGVector<float64_t>& result) const;

	// overloaded from base class
	virtual float64_t kernel(index_t idx_a, index_t idx_b) const;
	virtual SGMatrix<float64_t> dx_dx_dy(index_t idx_a, index_t idx_b) const;
	virtual float64_t dx_dx_dy_dy_sum(index_t idx_a, index_t idx_b) const;
	virtual SGMatrix<float64_t> dx_dy(index_t idx_a, index_t idx_b) const;
	virtual SGMatrix<float64_t> dx_dy_all() const;
	virtual SGVector<float64_t> dx(index_t a, index_t idx_b) const;
	virtual SGVector<float64_t> dx_dx(index_t a, index_t idx_b) const;
	virtual SGMatrix<float64_t> dx_i_dx_i_dx_j(index_t a, index_t idx_b) const;
	virtual SGMatrix<float64_t> dx_i_dx_j(index_t a, index_t idx_b) const;
	virtual SGMatrix<float64_t> dx_dx_dy_dy(index_t idx_a, index_t idx_b) const;
	virtual SGMatrix<float64_t> dx_i_dx_j_dx_k_dot_vec(index_t idx_a, index_t idx_b, const SGVector<float64_t>& vec) const;
	virtual SGMatrix<float64_t> dx_i_dx_j_dx_k_dx_k_dot_vec(index_t idx_a, index_t idx_b, const SGVector<float64_t>& vec) const;

	// nystrom parts
	virtual float64_t difference_component(index_t idx_a, index_t idx_b, index_t i) const;
	virtual float64_t dx_dy_component(const index_t idx_a, index_t idx_b, index_t i, index_t j) const;
	virtual float64_t dx_dx_dy_dy_component(index_t idx_a, index_t idx_b, index_t i, index_t j) const;
	virtual float64_t dx_dx_dy_component(index_t idx_a, index_t idx_b, index_t i, index_t j) const;
	virtual float64_t dx_component(index_t idx_a, index_t idx_b, index_t i) const;
	virtual float64_t dx_dx_component(index_t idx_a, index_t idx_b, index_t i) const;
	virtual SGVector<float64_t> dx_i_dx_i_dx_j_component(index_t idx_a, index_t idx_b, index_t i) const;
	virtual SGVector<float64_t> dx_i_dx_j_component(index_t idx_a, index_t idx_b, index_t i) const;

	virtual void precompute();

protected:
	float64_t m_sigma;

	// precomputed quantities
	SGMatrix<float64_t> m_sq_difference_norms;
	SGMatrix<float64_t> m_differences;
};
};
};

}
#endif // KERNEL_EXP_FAMILY_KERNEL_GAUSSIAN__
