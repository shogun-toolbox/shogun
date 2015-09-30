/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012-2013 Heiko Strathmann
 * Written (w) 2014 Soumyajit De
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
 */

#ifndef KERNEL_SELECTION_H_
#define KERNEL_SELECTION_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>

namespace shogun
{

class CKernelTwoSampleTest;
class CKernel;

/** @brief Base class for kernel selection for kernel two-sample test
 * statistic implementations (e.g. MMD).
 * Provides abstract methods for selecting kernels and computing criteria or
 * kernel weights for the implemented method. In order to implement new methods
 * for kernel selection, simply write a new implementation of this class.
 */
class CKernelSelection: public CSGObject
{
public:
	/** Default constructor */
	CKernelSelection();

	/** Constructor that initialises the underlying CKernelTwoSampleTest instance
	 *
	 * @param estimator CKernelTwoSampleTest instance to use.
	 */
	CKernelSelection(CKernelTwoSampleTest* estimator);

	/** Destructor */
	virtual ~CKernelSelection();

	/** If the the implemented method selects a single kernel, this computes
	 * criteria for all underlying kernels. If the method selects combined
	 * kernels, this method returns weights for the baseline kernels
	 *
	 * @return vector with criteria or kernel weights
	 */
	virtual SGVector<float64_t> compute_measures()=0;

	/** Abstract method that performs kernel selection on the base of the
	 * compute_measures() method and returns the selected kernel which is
	 * either a single or a combined one (with weights set)
	 *
	 * @return selected kernel (SG_REF'ed)
	 */
	virtual CKernel* select_kernel()=0;

	/** @param estimator the underlying CKernelTwoSampleTest instance */
	void set_estimator(CKernelTwoSampleTest* estimator);

	/** @return the underlying CKernelTwoSampleTest instance */
	CKernelTwoSampleTest* get_estimator() const;

	/** @return name of the SGSerializable */
	virtual const char* get_name() const
	{
		return "KernelSelection";
	}

private:
	/** Register parameters and initialize with default */
	void init();

protected:
	/** Underlying kernel two-sample test instance */
	CKernelTwoSampleTest* m_estimator;
};

}

#endif /* KERNEL_SELECTION_H_ */
