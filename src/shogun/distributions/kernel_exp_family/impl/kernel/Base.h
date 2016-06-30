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

#ifndef KERNEL_EXP_FAMILY_KERNEL_BASE__
#define KERNEL_EXP_FAMILY_KERNEL_BASE__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{
namespace kernel_exp_family_impl
{
namespace kernel
{

#define NOTIMPLEMENTED SG_SERROR("Kernel function not implemented yet!.\n");

class Base
{
public :
	Base();
	virtual ~Base() {};

	virtual void set_rhs(SGMatrix<float64_t> rhs);
	virtual void set_rhs(SGVector<float64_t> rhs);
	virtual void set_lhs(SGMatrix<float64_t> lhs);
	virtual void set_lhs(SGVector<float64_t> lhs);
	index_t get_num_dimensions() const;
	index_t get_num_lhs() const;
	index_t get_num_rhs() const;

	virtual float64_t kernel(index_t idx_a, index_t idx_b) const=0;
	virtual SGMatrix<float64_t> dx_dx_dy(index_t idx_a, index_t idx_b) const=0;
	virtual float64_t dx_dx_dy_dy_sum(index_t idx_a, index_t idx_b) const=0;
	virtual SGMatrix<float64_t> dx_dy(index_t idx_a, index_t idx_b) const=0;
	virtual SGMatrix<float64_t> dx_dy_all() const=0;

	virtual SGVector<float64_t> dx(index_t a, index_t idx_b) const=0;
	virtual SGVector<float64_t> dx_dx(index_t a, index_t idx_b) const=0;
	virtual SGMatrix<float64_t> dx_i_dx_i_dx_j(index_t a, index_t idx_b) const=0;
	virtual SGMatrix<float64_t> dx_i_dx_j(index_t a, index_t idx_b) const=0;

	// old develop code
	virtual SGMatrix<float64_t> dx_dx_dy_dy(index_t idx_a, index_t idx_b) const=0;

	// nystrom parts
	virtual float64_t difference_component(index_t idx_a, index_t idx_b, index_t i) const=0;
	virtual float64_t dx_dy_component(const index_t idx_a, index_t idx_b, index_t i, index_t j) const=0;
	virtual float64_t dx_dx_dy_dy_component(index_t idx_a, index_t idx_b, index_t i, index_t j) const=0;
	virtual float64_t dx_dx_dy_component(index_t idx_a, index_t idx_b, index_t i, index_t j) const=0;
	virtual float64_t dx_component(index_t idx_a, index_t idx_b, index_t i) const=0;
	virtual float64_t dx_dx_component(index_t idx_a, index_t idx_b, index_t i) const=0;
	virtual SGVector<float64_t> dx_i_dx_i_dx_j_component(index_t idx_a, index_t idx_b, index_t i) const=0;
	virtual SGVector<float64_t> dx_i_dx_j_component(index_t idx_a, index_t idx_b, index_t i) const=0;

	virtual void precompute() {};


protected:
	SGMatrix<float64_t> m_lhs;
	SGMatrix<float64_t> m_rhs;
};
};
};

}
#endif // KERNEL_EXP_FAMILY_KERNEL_BASE__
