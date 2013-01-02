/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#ifndef __MMDKERNELSELECTIONCOMBMAXL2_H_
#define __MMDKERNELSELECTIONCOMBMAXL2_H_

#include <shogun/statistics/MMDKernelSelection.h>
#include <shogun/statistics/MMDKernelSelectionComb.h>

namespace shogun
{

class CMMDKernelSelectionCombMaxL2: public CMMDKernelSelectionComb
{
public:

	/** Default constructor */
	CMMDKernelSelectionCombMaxL2();

	/** TODO
	 * Constructor that initialises the underlying MMD instance
	 *
	 * @param mmd MMD instance to use. Has to be an MMD based kernel two-sample
	 * test. Currently: linear or quadratic time MMD.
	 */
	CMMDKernelSelectionCombMaxL2(CKernelTwoSampleTestStatistic* mmd,
			float64_t lambda=10E-5);

	/** Destructor */
	virtual ~CMMDKernelSelectionCombMaxL2();

	/** TODO */
	virtual SGVector<float64_t> compute_measures();

	/** @return name of the SGSerializable */
	const char* get_name() const { return "MMDKernelSelectionCombMaxL2"; }
};

}

#endif /* __MMDKERNELSELECTIONCOMBMAXL2_H_ */
