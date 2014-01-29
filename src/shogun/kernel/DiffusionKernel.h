/** The Shogun Machine Learning Toolbox
 *  Copyright (c) 2014, The Shogun-Team
 * All rights reserved.
 *
 * Distributed under the BSD 2-clause license:
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _DIFFUSION_H___
#define _DIFFUSION_H___

#include <shogun/kernel/DotKernel.h>

namespace shogun
{
class CDiffusionKernel: public CDotKernel
{
	public:
		CDiffusionKernel();

		CDiffusionKernel(SGVector<float64_t> betas,
				SGVector<index_t> alphabet_sizes);

		virtual ~CDiffusionKernel();

		virtual bool init(CFeatures* l, CFeatures* r);

		void set_betas(SGVector<float64_t> betas);

		/** @return kernel type */
		virtual EKernelType get_kernel_type() { return K_DIFFUSION; }

		/** @return name of SG_SERIALIZABLE */
		virtual const char* get_name() const { return "DiffusionKernel"; }

	protected:
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

		void precompute_decay_numbers();

	private:
		/** initializes and registers parameters */
		void init();

	protected:
		SGVector<float64_t> m_betas;
		SGVector<index_t> m_alphabet_sizes;

		SGVector<float64_t> m_decay_numbers;
};
}
#endif /* _GAUSSIANKERNEL_H__ */
