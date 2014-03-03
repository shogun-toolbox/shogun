/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#ifndef __MMDKERNELSELECTIONCOMBOPT_H_
#define __MMDKERNELSELECTIONCOMBOPT_H_

#include <shogun/statistics/MMDKernelSelectionComb.h>

namespace shogun
{

class CLinearTimeMMD;

/** @brief Implementation of optimal kernel selection for combined kernel.
 * This class selects a combination of baseline kernels that maximises the
 * ratio of the MMD and its standard deviation for a combined kernel. This
 * boils down to solve the convex program
 * \f[
 * \min_\beta \{\beta^T (Q+\lambda_m) \beta \quad \text{s.t.}\quad \beta^T \eta=1, \beta\succeq 0\},
 * \f]
 * where \f$\eta\f$ is a vector whose elements are the MMDs of the baseline
 * kernels and \f$Q\f$ is a linear time estimate of the covariance of \f$\eta\f$.
 *
 * This only works for the CLinearTimeMMD statistic. *
 * IMPORTANT: The kernel has to be selected on different data than the two-sample
 * test is performed on.
 *
 * The method is described in
 * Gretton, A., Sriperumbudur, B., Sejdinovic, D., Strathmann, H.,
 * Balakrishnan, S., Pontil, M., & Fukumizu, K. (2012).
 * Optimal kernel choice for large-scale two-sample tests.
 * Advances in Neural Information Processing Systems.
 */
class CMMDKernelSelectionCombOpt: public CMMDKernelSelectionComb
{
public:

	/** Default constructor */
	CMMDKernelSelectionCombOpt();

	/** Constructor that initialises the underlying MMD instance
	 *
	 * @param mmd linear time mmd MMD instance to use.
	 * @param lambda ridge that is added to standard deviation, a sensible value
	 * is 10E-5 which is the default
	 */
	CMMDKernelSelectionCombOpt(CKernelTwoSampleTest* mmd,
			float64_t lambda=10E-5);

	/** Destructor */
	virtual ~CMMDKernelSelectionCombOpt();

#ifdef HAVE_LAPACK
	/** Computes optimal kernel weights using the ratio of the squared MMD by its
	 * standard deviation as a criterion, where both expressions are estimated
	 * in linear time.
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
	const char* get_name() const { return "MMDKernelSelectionCombOpt"; }

private:
	/** Initializer */
	void init();

protected:
	/** Ridge that is added to the diagonal of the Q matrix in the optimization
	 * problem */
	float64_t m_lambda;
};

}

#endif /* __MMDKERNELSELECTIONCOMBOPT_H_ */
