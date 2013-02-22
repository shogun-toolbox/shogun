/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#ifndef __MMDKERNELSELECTIONMEDIAN_H_
#define __MMDKERNELSELECTIONMEDIAN_H_

#include <shogun/statistics/MMDKernelSelection.h>

namespace shogun
{

class CMMDKernelSelectionMedian: public CMMDKernelSelection
{
public:

	/** Default constructor */
	CMMDKernelSelectionMedian();

	/** Constructor that initialises the underlying MMD instance
	 *
	 * @param mmd MMD instance to use. Has to be an MMD based kernel two-sample
	 * test.
	 * @param num_data_distance TODO description
	 * TODO
	 */
	CMMDKernelSelectionMedian(CKernelTwoSampleTestStatistic* mmd,
			index_t num_data_distance=1000);

	/** Destructor */
	virtual ~CMMDKernelSelectionMedian();

	/** TODO */
	virtual SGVector<float64_t> compute_measures();

	/** TODO */
	virtual CKernel* select_kernel();

	/** @return name of the SGSerializable */
	const char* get_name() const { return "MMDKernelSelectionMedian"; }

private:
	/* initialises and registers member variables */
	void init();

protected:
	/** maximum number of data to be used for median distance computation */
	index_t m_num_data_distance;
};

}

#endif /* __MMDKERNELSELECTIONMEDIAN_H_ */
