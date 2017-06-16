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

#ifndef KERNEL_EXP_FAMILY_IMPL_NYSTROM_D__
#define KERNEL_EXP_FAMILY_IMPL_NYSTROM_D__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include "Nystrom.h"

namespace shogun
{

namespace kernel_exp_family_impl
{
class NystromD : public Nystrom
{
public :
	NystromD(SGMatrix<float64_t> data, SGMatrix<bool> basis_mask,
			kernel::Base* kernel, float64_t lambda, float64_t lambda_l2=0.0);

	NystromD(SGMatrix<float64_t> data, SGMatrix<float64_t> basis,
			SGMatrix<bool> basis_mask,
			kernel::Base* kernel, float64_t lambda, float64_t lambda_l2=0.0);

	void set_basis_inds_from_mask(const SGMatrix<bool>& basis_mask);

	// overloaded from Full base class
	virtual SGVector<float64_t> compute_h() const;

	// overloaded from base Nystrom class for sub-sampling dimensions
	virtual bool basis_is_subsampled_data() const;
	virtual index_t get_system_size() const;
	virtual SGMatrix<float64_t> subsample_G_mm_from_G_mn(const SGMatrix<float64_t>& G_mn) const;
	virtual SGMatrix<float64_t> compute_G_mn() const;
	virtual SGMatrix<float64_t> compute_G_mm(); // TODO this should be const!
	SGVector<float64_t> solve_system(const SGMatrix<float64_t>& system_matrix,
			const SGVector<float64_t>& system_vector) const;

protected:
};
};

}
#endif // KERNEL_EXP_FAMILY_IMPL_NYSTROM_D__
