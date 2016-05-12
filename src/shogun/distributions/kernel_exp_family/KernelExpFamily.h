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

#ifndef KERNEL_EXP_FAMILY__
#define KERNEL_EXP_FAMILY__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>

namespace shogun
{

template<class ST> class CDenseFeatures;
class KernelExpFamilyImpl;

class CKernelExpFamily : public CSGObject
{
public:
	CKernelExpFamily();
	CKernelExpFamily(SGMatrix<float64_t> data,
			float64_t sigma, float64_t lambda);
	virtual ~CKernelExpFamily();

	void fit();
	float64_t log_pdf(SGVector<float64_t> x);
	SGVector<float64_t> log_pdf_multiple(SGMatrix<float64_t> X);

	virtual const char* get_name() const { return "KernelExpFamily"; }

	SGVector<float64_t> get_alpha_beta();

protected:
	KernelExpFamilyImpl* m_impl;
};

}
#endif // KERNEL_EXP_FAMILY__
