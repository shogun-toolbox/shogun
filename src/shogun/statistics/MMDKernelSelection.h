/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#ifndef __MMDKERNELSELECTION_H_
#define __MMDKERNELSELECTION_H_

#include <base/SGObject.h>

namespace shogun
{

class CKernelTwoSampleTestStatistic;
class CKernel;

/** @brief Base class for kernel selection for MMD-based two-sample test
 * statistic implementations (e.g. MMD).
 * Provides abstract methods for selecting kernels and computing criteria or
 * kernel weights for the implemented method. In order to implement new methods
 * for kernel selection, simply write a new implementation of this class.
 *
 * Kernel selection works this way: One passes an instance of CCombinedKernel
 * to the MMD statistic and appends all kernels that should be considered.
 * Depending on the type of kernel selection implementation, a single one or
 * a combination of those baseline kernels is selected and returned to the user.
 * This kernel can then be passed to the MMD instance to perform a test.
 *
 */
class CMMDKernelSelection: public CSGObject
{
public:

	/** Default constructor */
	CMMDKernelSelection();

	/** Constructor that initialises the underlying MMD instance
	 *
	 * @param mmd MMD instance to use. Has to be an MMD based kernel two-sample
	 * test. Currently: linear or quadratic time MMD.
	 */
	CMMDKernelSelection(CKernelTwoSampleTestStatistic* mmd);

	/** Destructor */
	virtual ~CMMDKernelSelection();

	/** If the the implemented method selects a single kernel, this computes
	 * criteria for all underlying kernels. If the method selects combined
	 * kernels, this method returns weights for the baseline kernels
	 *
	 * @return vector with criteria or kernel weights
	 */
	virtual SGVector<float64_t> compute_measures()=0;

	/** Performs kernel selection on the base of the compute_measures() method
	 * and returns the selected kernel which is either a single or a combined
	 * one (with weights set)
	 *
	 * @return selected kernel (SG_REF'ed)
	 */
	virtual CKernel* select_kernel();

	/** @return name of the SGSerializable */
	const char* get_name() const=0;

private:

	/** Initializer */
	void init();

protected:
	/** Underlying MMD instance */
	CKernelTwoSampleTestStatistic* m_mmd;
};

}

#endif /* __MMDKERNELSELECTION_H_ */
