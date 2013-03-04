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

#include <shogun/lib/config.h>
#include <shogun/statistics/MMDKernelSelection.h>
#include <shogun/statistics/MMDKernelSelectionComb.h>

namespace shogun
{

/** @brief Implementation of maximum MMD kernel selection for combined kernel.
 * This class selects a combination of baseline kernels that maximises the
 * the MMD for a combined kernel based on a L2-regularization approach. This
 * boils down to solve the convex program
 * \f[
 * \min_\beta \{\beta^T \beta \quad \text{s.t.}\quad \beta^T \eta=1, \beta\succeq 0\},
 * \f]
 * where \f$\eta\f$ is a vector whose elements are the MMDs of the baseline
 * kernels.
 *
 * This is meant to work for the CQuadraticTimeMMD statistic.
 * Optimal weight selecton for CLinearTimeMMD can be found in
 * CMMDKernelSelectionCombOpt.
 *
 * The method is described in
 * Gretton, A., Sriperumbudur, B., Sejdinovic, D., Strathmann, H.,
 * Balakrishnan, S., Pontil, M., & Fukumizu, K. (2012).
 * Optimal kernel choice for large-scale two-sample tests.
 * Advances in Neural Information Processing Systems.
 */
class CMMDKernelSelectionCombMaxL2: public CMMDKernelSelectionComb
{
public:

	/** Default constructor */
	CMMDKernelSelectionCombMaxL2();

	/** Constructor that initialises the underlying MMD instance
	 *
	 * @param mmd MMD instance to use. Has to be an MMD based kernel two-sample
	 * test. Currently: linear or quadratic time MMD.
	 */
	CMMDKernelSelectionCombMaxL2(CKernelTwoSampleTestStatistic* mmd);

	/** Destructor */
	virtual ~CMMDKernelSelectionCombMaxL2();

#ifdef HAVE_LAPACK
	/** Computes kernel weights which maximise the MMD of the underlying
	 * combined kernel using L2-regularization.
	 *
	 * This boils down to solving a convex program which is quadratic in the
	 * number of kernels. See class description.
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
	const char* get_name() const { return "MMDKernelSelectionCombMaxL2"; }
};

}

#endif /* __MMDKERNELSELECTIONCOMBMAXL2_H_ */
