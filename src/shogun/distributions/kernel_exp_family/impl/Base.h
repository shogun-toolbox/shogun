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

#ifndef KERNEL_EXP_FAMILY_IMPL_BASE__
#define KERNEL_EXP_FAMILY_IMPL_BASE__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <utility>
#include <memory>

#include "kernel/Base.h"

namespace shogun
{

namespace kernel_exp_family_impl
{

namespace kernel
{
class Base;
};

class Base
{
public :
	Base(SGMatrix<float64_t> data, std::shared_ptr<kernel::Base> kernel, float64_t lambda,
			bool init_base_and_data=true);
	virtual ~Base();

	void set_data(SGMatrix<float64_t> X);
	void set_data(SGVector<float64_t> x);
	void reset_data();
	bool is_basis_equal_data() const;

	virtual void fit()=0;

	float64_t score() const;

	SGVector<float64_t> get_beta() const { return m_beta; }
	void set_beta(SGVector<float64_t> beta);

	index_t get_num_dimensions() const;
	index_t get_num_basis() const;
	index_t get_num_data() const;

	virtual float64_t log_pdf(index_t idx_test) const=0;
	virtual SGVector<float64_t> grad(index_t idx_test) const=0;
	virtual SGMatrix<float64_t> hessian(index_t idx_test) const=0;
	virtual SGVector<float64_t> hessian_diag(index_t idx_test) const=0;

	virtual SGVector<float64_t> log_pdf() const;
	virtual SGMatrix<float64_t> grad() const;

protected:
	virtual void set_basis_and_data(SGMatrix<float64_t> basis,
			SGMatrix<float64_t> data);

	std::shared_ptr<kernel::Base> m_kernel;
	float64_t m_lambda;

	SGVector<float64_t> m_beta;
	SGMatrix<float64_t> m_basis;

	SGMatrix<float64_t> m_data;
};
};

}
#endif // KERNEL_EXP_FAMILY_IMPL_BASE__
