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
	Base(SGMatrix<float64_t> data, kernel::Base* kernel, float64_t lambda);
	virtual ~Base();

	// for evaluation
	void set_test_data(SGMatrix<float64_t> X);
	void set_test_data(SGVector<float64_t> x);

	void fit();

	float64_t log_pdf(SGVector<float64_t> x);
	SGVector<float64_t> log_pdf(SGMatrix<float64_t> X);

	SGVector<float64_t> grad(SGVector<float64_t> x);
	SGMatrix<float64_t> hessian(SGVector<float64_t> x);

	SGVector<float64_t> get_alpha_beta() const { return m_alpha_beta; }

	index_t get_num_dimensions() const;
	index_t get_num_lhs() const;
	index_t get_num_rhs() const;

	virtual std::pair<SGMatrix<float64_t>, SGVector<float64_t>> build_system() const=0;
	virtual float64_t log_pdf(index_t idx_test) const=0;
	virtual SGVector<float64_t> grad(index_t idx_test) const=0;
	virtual SGMatrix<float64_t> hessian(index_t idx_test) const=0;

protected:
	virtual void solve_and_store(const SGMatrix<float64_t>& A, const SGVector<float64_t>& b);

	kernel::Base* m_kernel;
	float64_t m_lambda;

	SGVector<float64_t> m_alpha_beta;
};
};

}
#endif // KERNEL_EXP_FAMILY_IMPL_BASE__
