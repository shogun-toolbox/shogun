/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __MMDKERNELSELECTIONOPTCOMB_H_
#define __MMDKERNELSELECTIONOPTCOMB_H_

#include <shogun/statistics/MMDKernelSelection.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/external/libqp.h>

namespace shogun
{

class CLinearTimeMMD;

class CMMDKernelSelectionOptComb: public CMMDKernelSelection
{
public:

	/** Default constructor */
	CMMDKernelSelectionOptComb();

	/** Constructor that initialises the underlying MMD instance
	 *
	 * @param linear time mmd MMD instance to use.
	 * @param lamda ridge that is added to standard deviation
	 * @param combine_kernels switches between two modes: using a single best
	 * kernel and using the best weights for a convex combination of kernels
	 */
	CMMDKernelSelectionOptComb(CKernelTwoSampleTestStatistic* mmd,
			float64_t lambda);

	/** Destructor */
	virtual ~CMMDKernelSelectionOptComb();

	/** Overrides the superclass method and throws an error. Use select_kernel()
	 * instead (possible to access subkernel weights from it).
	 *
	 * @return nothing
	 */
	virtual SGVector<float64_t> compute_measures();

	/** Returns a combined kernel version of all underlying base kernels,
	 * weighted in an optimal way. Throws an error if LAPACK is not installed.
	 *
	 * @return combined kernel with all underlying kernels and optimal weights
	 */
	virtual CKernel* select_kernel();


	/** @return name of the SGSerializable */
	const char* get_name() const { return "MMDKernelSelectionOptComb"; }

protected:
	virtual float64_t compute_measure(CKernel* kernel);


private:
	/** Initializer */
	void init();

#ifdef HAVE_LAPACK
	/** return pointer to i-th column of m_Q. Helper for libqp */
	static const float64_t* get_Q_col(uint32_t i);

	/** helper functions that prints current state */
	static void print_state(libqp_state_T state);
#endif

protected:
	/** Ridge that is added to the diagonal of the Q matrix in the optimization
	 * problem */
	float64_t m_lambda;

#ifdef HAVE_LAPACK
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

#endif /* __MMDKERNELSELECTIONOPTCOMB_H_ */
