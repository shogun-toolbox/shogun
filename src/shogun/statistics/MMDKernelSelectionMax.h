/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#ifndef __MMDKERNELSELECTIONMAX_H_
#define __MMDKERNELSELECTIONMAX_H_

#include <shogun/statistics/MMDKernelSelection.h>

namespace shogun
{

/** @brief Kernel selection class that selects the single kernel that maximises
 * the MMD statistic. Works for CQuadraticTimeMMD and CLinearTimeMMD. This leads
 * to a heuristic that is better than the standard median heuristic for
 * Gaussian kernels. However, it comes with no guarantees.
 *
 * Optimal selection of single kernels can be found in the class
 * CMMDKernelSelectionOpt
 *
 * This method was first described in
 * Sriperumbudur, B., Fukumizu, K., Gretton, A., Lanckriet, G. R. G.,
 * & Schoelkopf, B.
 * Kernel choice and classifiability for RKHS embeddings of probability
 * distributions. Advances in Neural Information Processing Systems (2009).
 */
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
	CMMDKernelSelectionMax(CKernelTwoSampleTest* mmd);

	/** Destructor */
	virtual ~CMMDKernelSelectionMax();

	/** @return vector the MMD of all single baseline kernels */
	virtual SGVector<float64_t> compute_measures();

	/** @return name of the SGSerializable */
	const char* get_name() const { return "MMDKernelSelectionMax"; }
};

}

#endif /* __MMDKERNELSELECTIONMAX_H_ */
