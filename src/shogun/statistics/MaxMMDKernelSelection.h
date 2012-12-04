/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __MAXMMDKERNELSELECTION_H_
#define __MAXMMDKERNELSELECTION_H_

#include <shogun/statistics/MMDKernelSelection.h>

namespace shogun
{

class CMaxMMDKernelSelection: public CMMDKernelSelection
{
public:

	/** Default constructor */
	CMaxMMDKernelSelection();

	/** Constructor that initialises the underlying MMD instance
	 *
	 * @param mmd MMD instance to use. Has to be an MMD based kernel two-sample
	 * test. Currently: linear or quadratic time MMD.
	 */
	CMaxMMDKernelSelection(CKernelTwoSampleTestStatistic* mmd);

	/** Destructor */
	virtual ~CMaxMMDKernelSelection();

	/** @return name of the SGSerializable */
	const char* get_name() const { return "MaxMMDKernelSelection"; }

protected:
	virtual float64_t compute_measure(CKernel* kernel);
};

}

#endif /* __MAXMMDKERNELSELECTION_H_ */
