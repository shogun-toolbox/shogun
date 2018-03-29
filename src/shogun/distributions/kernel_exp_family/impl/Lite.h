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

#ifndef KERNEL_EXP_FAMILY_IMPL_LITE__
#define KERNEL_EXP_FAMILY_IMPL_LITE__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include <memory>

#include "Base.h"
#include "Full.h"
#include "Nystrom.h"

namespace shogun
{

namespace kernel_exp_family_impl
{
class Lite : public Nystrom
{
public :
	Lite(SGMatrix<float64_t> data, SGMatrix<float64_t> basis,
			std::shared_ptr<kernel::Base> kernel, float64_t lambda,
			float64_t lambda_l2=0.0, bool init_base_and_data=true);

	Lite(SGMatrix<float64_t> data, SGVector<index_t> basis_inds,
			std::shared_ptr<kernel::Base> kernel, float64_t lambda,
			float64_t lambda_l2=0.0, bool init_base_and_data=true);

	Lite(SGMatrix<float64_t> data, index_t num_subsample_basis,
			std::shared_ptr<kernel::Base> kernel, float64_t lambda,
			float64_t lambda_l2=0.0, bool init_base_and_data=true);

	virtual ~Lite() {};

	virtual index_t get_system_size() const;

	virtual SGMatrix<float64_t> subsample_G_mm_from_G_mn(const SGMatrix<float64_t>& G_mn) const;
	virtual bool can_subsample_G_mm_from_G_mn() const;
	virtual SGMatrix<float64_t> compute_G_mn() const;
	virtual SGMatrix<float64_t> compute_G_mm() const;
	virtual SGVector<float64_t> compute_h() const;

protected:

	float64_t m_lambda_l2;
	SGVector<index_t> m_basis_inds;
};
};

}
#endif // KERNEL_EXP_FAMILY_IMPL_LITE__
