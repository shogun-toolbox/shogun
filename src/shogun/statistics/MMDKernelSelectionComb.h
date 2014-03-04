/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#ifndef __MMDKERNELSELECTIONCOMB_H_
#define __MMDKERNELSELECTIONCOMB_H_

#include <shogun/lib/config.h>

#include <shogun/statistics/MMDKernelSelection.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/external/libqp.h>

namespace shogun
{

class CLinearTimeMMD;

/** @brief Base class for kernel selection of combined kernels. Given an MMD
 * instance whose underlying kernel is a combined one, this class provides an
 * interface to select weights of this combined kernel.
 */
class CMMDKernelSelectionComb: public CMMDKernelSelection
{
public:

	/** Default constructor */
	CMMDKernelSelectionComb();

	/** Constructor that initialises the underlying MMD instance. Currently,
	 * only the linear time MMD is supported
	 *
	 * @param mmd MMD instance to use
	 */
	CMMDKernelSelectionComb(CKernelTwoSampleTest* mmd);

	/** Destructor */
	virtual ~CMMDKernelSelectionComb();

#ifdef HAVE_LAPACK
	/** Abstract method that computes weights of the selected combined kernel.
	 *
	 * @return weights of the selected kernel
	 */
	virtual SGVector<float64_t> compute_measures()=0;
#else
	/** Abstract method that computes weights of the selected combined kernel.
	 * LAPACK needs to be installed for this method to work. Please install!
	 *
	 * @return Throws an error
	 */
	virtual SGVector<float64_t> compute_measures();
#endif

	/** @return computes weights for the underlying kernel, sets them to it, and
	 * returns it (SG_REF'ed)
	 *
	 * @return underlying kernel with weights set
	 */
	virtual CKernel* select_kernel();

	/** @return name of the SGSerializable */
	const char* get_name() const=0;

protected:
	/** Solves the quadratic program
	 * \f[
	 * \min_\beta \{\beta^T Q \beta \quad \text{s.t.}\quad \beta^T \eta=1, \beta\succeq 0\},
	 * \f]
	 * where \f$\eta\f$ is a given parameter and \f$Q\f$ is the m_Q member.
	 *
	 * Note that at least one element is assumed \f$\eta\f$ has to be positive.
	 *
	 * @param mmds values that will be put into \f$\eta\f$. At least one element
	 * is assumed to be positive
	 * @return result of optimization \f$\beta\f$
	 */
	virtual SGVector<float64_t> solve_optimization(SGVector<float64_t> mmds);

#ifdef HAVE_LAPACK
	/** return pointer to i-th column of m_Q. Helper for libqp */
	static const float64_t* get_Q_col(uint32_t i);

	/** helper function that prints current state */
	static void print_state(libqp_state_T state);

	/** maximum number of iterations of qp solver */
	index_t m_opt_max_iterations;

	/** stopping accuracy of qp solver */
	float64_t m_opt_epsilon;

	/** low cut for weights, if weights are under this value, are set to zero */
	float64_t m_opt_low_cut;

	/** matrix for selection of kernel weights (static because of libqp) */
	static SGMatrix<float64_t> m_Q;
#endif

private:
	/** initializer */
	void init();
};

}

#endif /* __MMDKERNELSELECTIONCOMB_H_ */
