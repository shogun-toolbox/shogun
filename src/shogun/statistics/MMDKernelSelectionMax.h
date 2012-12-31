/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __MMDKERNELSELECTIONMAXMMD_H_
#define __MMDKERNELSELECTIONMAXMMD_H_

#include <shogun/statistics/MMDKernelSelection.h>

namespace shogun
{

class CMMDKernelSelectionMax: public CMMDKernelSelection
{
public:

	/** Default constructor */
	CMMDKernelSelectionMax();

	/** Constructor that initialises the underlying MMD instance
	 *
	 * @param mmd MMD instance to use. Has to be an MMD based kernel two-sample
	 * test. Currently: linear or quadratic time MMD.
	 */
	CMMDKernelSelectionMax(CKernelTwoSampleTestStatistic* mmd);

	/** Destructor */
	virtual ~CMMDKernelSelectionMax();

	/** TODO */
	virtual SGVector<float64_t> compute_measures();

	/** @return name of the SGSerializable */
	const char* get_name() const { return "MMDKernelSelectionMax"; }

};

}

#endif /* __MMDKERNELSELECTIONMAXMMD_H_ */
