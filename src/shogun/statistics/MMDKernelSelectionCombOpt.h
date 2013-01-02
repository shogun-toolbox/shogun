/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __MMDKERNELSELECTIONCOMBOPT_H_
#define __MMDKERNELSELECTIONCOMBOPT_H_

#include <shogun/statistics/MMDKernelSelectionComb.h>

namespace shogun
{

class CLinearTimeMMD;

class CMMDKernelSelectionCombOpt: public CMMDKernelSelectionComb
{
public:

	/** Default constructor */
	CMMDKernelSelectionCombOpt();

	/** TODO
	 * Constructor that initialises the underlying MMD instance
	 *
	 * @param linear time mmd MMD instance to use.
	 * @param lamda ridge that is added to standard deviation, a sensible value
	 * is 10E-5 which is the default
	 * @param combine_kernels switches between two modes: using a single best
	 * kernel and using the best weights for a convex combination of kernels
	 */
	CMMDKernelSelectionCombOpt(CKernelTwoSampleTestStatistic* mmd,
			float64_t lambda=10E-5);

	/** Destructor */
	virtual ~CMMDKernelSelectionCombOpt();

#ifdef HAVE_LAPACK
	/** TODO
	 * Computes optimal kernel weights using the ratio of the squared MMD by its
	 * standard deviation as a criterion, i.e.
	 * \f[
	 * \frac{\text{MMD}_l^2[\mathcal{F},X,Y]}{\sigma_l}
	 * \f]
	 * where both expressions are estimated in linear time.
	 * This comes down to solving a convex program which is quadratic in the
	 * number of kernels.
	 *
	 * SHOGUN has to be compiled with LAPACK to make this available. See
	 * set_opt* methods for optimization parameters.
	 *
	 * IMPORTANT: Kernel weights have to be learned on different data than is
	 * used for testing/evaluation!
	 */
	virtual SGVector<float64_t> compute_measures();
#endif

	/** @return name of the SGSerializable */
	const char* get_name() const { return "MMDKernelSelectionCombOpt"; }
};

}

#endif /* __MMDKERNELSELECTIONCOMBOPT_H_ */
