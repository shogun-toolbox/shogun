/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#ifndef __MMDKERNELSELECTIONOPTSINGLE_H_
#define __MMDKERNELSELECTIONOPTSINGLE_H_

#include <shogun/lib/config.h>
#include <shogun/statistics/MMDKernelSelection.h>

namespace shogun
{

class CLinearTimeMMD;

/** @brief Implements optimal kernel selection for single kernels.
 * Given a number of baseline kernels, this method selects the one that
 * minimizes the type II error for a given type I error for a two-sample test.
 * This only works for the CLinearTimeMMD statistic.
 *
 * The idea is to maximise the ratio of MMD and its standard deviation.
 *
 * IMPORTANT: The kernel has to be selected on different data than the two-sample
 * test is performed on.
 *
 * Described in
 * Gretton, A., Sriperumbudur, B., Sejdinovic, D., Strathmann, H.,
 * Balakrishnan, S., Pontil, M., & Fukumizu, K. (2012).
 * Optimal kernel choice for large-scale two-sample tests.
 * Advances in Neural Information Processing Systems.
 */
class CMMDKernelSelectionOpt: public CMMDKernelSelection
{
public:

	/** Default constructor */
	CMMDKernelSelectionOpt();

	/** Constructor that initialises the underlying MMD instance. Currently,
	 * only the linear time MMD is supported
	 *
	 * @param mmd MMD instance to use
	 * @param lambda ridge that is added to standard deviation in order to
	 * prevent division by zero. A sensivle value is for example 1E-5.
	 */
	CMMDKernelSelectionOpt(CKernelTwoSampleTest* mmd,
			float64_t lambda=10E-5);

	/** Destructor */
	virtual ~CMMDKernelSelectionOpt();

	/** Overwrites superclass method and ensures that all statistics are
	 * computed on the same data. Since linear time MMD is a streaming
	 * statistic, just computing all statistics one after another would use
	 * different data. This method makes sure that all kernels are used at once
	 *
	 * @return vector with kernel criterion values for all attached kernels
	 */
	virtual SGVector<float64_t> compute_measures();

	/** @return name of the SGSerializable */
	const char* get_name() const { return "MMDKernelSelectionOpt"; }

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
