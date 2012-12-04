/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __MMDKERNELSELECTIONOPTSINGLE_H_
#define __MMDKERNELSELECTIONOPTSINGLE_H_

#include <shogun/statistics/MMDKernelSelection.h>

namespace shogun
{

class CLinearTimeMMD;

class CMMDKernelSelectionOptSingle: public CMMDKernelSelection
{
public:

	/** Default constructor */
	CMMDKernelSelectionOptSingle();

	/** Constructor that initialises the underlying MMD instance. Currently,
	 * only the linear time MMD is developed
	 *
	 * @param mmd MMD instance to use
	 * @param lamda ridge that is added to standard deviation
	 */
	CMMDKernelSelectionOptSingle(CKernelTwoSampleTestStatistic* mmd,
			float64_t lambda);

	/** Destructor */
	virtual ~CMMDKernelSelectionOptSingle();

	/** @return name of the SGSerializable */
	const char* get_name() const { return "MMDKernelSelectionOptSingle"; }

protected:
	virtual float64_t compute_measure(CKernel* kernel);

private:
	/** Initializer */
	void init();

protected:
	/** Ridge that is added to the denumerator of the ratio of MMD and its
	 * standard deviation */
	float64_t m_lambda;
};

}

#endif /* __MMDKERNELSELECTIONOPTSINGLE_H_ */
