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

#include <statistics/MMDKernelSelection.h>

namespace shogun
{

/** @brief Implements MMD kernel selection for a number of Gaussian baseline
 * kernels via selecting the one with a bandwidth parameter that is closest to
 * the median of all pairwise distances in the underlying data. Therefore, it
 * only works for data to which a GaussianKernel can be applied, which are
 * grouped under the class CDotFeatures in SHOGUN.
 *
 * This method works reasonable if distinguishing characteristics of data are not
 * hidden at a different length-scale that the overall one. In addition it is
 * fast to compute. In other cases, it is a bad choice.
 *
 * Optimal selection of single kernels can be found in the class
 * CMMDKernelSelectionOpt
 *
 * Described among oher places in
 * Gretton, A., Borgwardt, K. M., Rasch, M. J., Schoelkopf, B., & Smola, A.
 * (2012).
 * A Kernel Two-Sample Test. Journal of Machine Learning Research, 13, 671-721.
 */
class CMMDKernelSelectionMedian: public CMMDKernelSelection
{
public:

	/** Default constructor */
	CMMDKernelSelectionMedian();

	/** Constructor that initialises the underlying MMD instance
	 *
	 * @param mmd MMD instance to use. Has to be an MMD based kernel two-sample
	 * test.
	 * @param num_data_distance Number of points that is used to compute the
	 * median distance on. Since the median is stable, this do need need to be
	 * all data, but a small subset is sufficient.
	 */
	CMMDKernelSelectionMedian(CKernelTwoSampleTestStatistic* mmd,
			index_t num_data_distance=1000);

	/** Destructor */
	virtual ~CMMDKernelSelectionMedian();

	/** @return Throws an error and shoold not be used */
	virtual SGVector<float64_t> compute_measures();

	/** Returns the baseline kernel whose bandwidth parameter is closest to the
	 * median of the pairwise distances of the underlyinf data
	 *
	 * @return selected kernel (SG_REF'ed)
	 */
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
