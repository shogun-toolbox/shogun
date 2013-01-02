/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __MMDKERNELSELECTIONCOMB_H_
#define __MMDKERNELSELECTIONCOMB_H_

#include <shogun/statistics/MMDKernelSelection.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/external/libqp.h>

namespace shogun
{

class CLinearTimeMMD;

class CMMDKernelSelectionComb: public CMMDKernelSelection
{
public:

	/** Default constructor */
	CMMDKernelSelectionComb();

	/** TODO
	 * Constructor that initialises the underlying MMD instance
	 *
	 * @param linear time mmd MMD instance to use.
	 * @param lamda ridge that is added to standard deviation, a sensible value
	 * is 10E-% which is the default
	 * @param combine_kernels switches between two modes: using a single best
	 * kernel and using the best weights for a convex combination of kernels
	 */
	CMMDKernelSelectionComb(CKernelTwoSampleTestStatistic* mmd,
			float64_t lambda=10E-5);

	/** Destructor */
	virtual ~CMMDKernelSelectionComb();

#ifdef HAVE_LAPACK
	/** TODO
	 */
	virtual SGVector<float64_t> compute_measures()=0;
#else
	virtual SGVector<float64_t> compute_measures();
#endif

	/* TODO */
	virtual CKernel* select_kernel();

	/** @return name of the SGSerializable */
	const char* get_name() const=0;

private:
	/** initializer */
	void init();

#ifdef HAVE_LAPACK
protected:
	/** TODO */
	virtual SGVector<float64_t> solve_optimization(SGVector<float64_t> mmds);

	/** return pointer to i-th column of m_Q. Helper for libqp */
	static const float64_t* get_Q_col(uint32_t i);

	/** helper function that prints current state */
	static void print_state(libqp_state_T state);

	/** Ridge that is added to the diagonal of the Q matrix in the optimization
	 * problem */
	float64_t m_lambda;

	/** maximum number of iterations of qp solver */
	index_t m_opt_max_iterations;

	/** stopping accuracy of qp solver */
	float64_t m_opt_epsilon;

	/** low cut for weights, if weights are under this value, are set to zero */
	float64_t m_opt_low_cut;

	/** matrix for selection of kernel weights (static because of libqp) */
	static SGMatrix<float64_t> m_Q;
#endif
};

}

#endif /* __MMDKERNELSELECTIONCOMB_H_ */
