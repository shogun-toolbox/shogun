/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#ifndef __KERNELINDEPENDENCESTSTATISTIC_H_
#define __KERNELINDEPENDENCESTSTATISTIC_H_

#include <shogun/statistics/IndependenceTestStatistic.h>

namespace shogun
{

class CFeatures;
class CKernel;

/** @brief Kernel independence test base class. Provides an interface for
 * performing an independence test. Given samples \f$Z=\{(x_i,y_i)\}_{i=1}^m\f$
 * from the joint distribution \f$\textbf{P}_{xy}\f$, does the joint
 * distribution factorize as \f$\textbf{P}_{xy}=\textbf{P}_x\textbf{P}_y\f$,
 * i.e. product of the marginals?
 *
 * The null-hypothesis says yes, i.e. no dependence, the alternative hypothesis
 * says no.
 *
 * In this class, this is done using a single kernel for each of the two sets
 * of samples
 *
 * The class also re-implements the sample_null() method. If the underlying
 * kernel is a custom one (precomputed), the rows and column of the kernel
 * matrix for p is permuted using subsets. The computation falls back to
 * CIndependenceTestStatistic::sample_null() otherwise.
 *
 * Abstract base class.
 */
class CKernelIndependenceTestStatistic: public CIndependenceTestStatistic
{
public:
	/** default constructor */
	CKernelIndependenceTestStatistic();

	/** Constructor
	 *
	 * @param p_and_q feature data. Is assumed to contain samples from both
	 * p and q. First all samples from p, then from index q_start all
	 * samples from q
	 *
	 * @param kernel_p kernel to use on samples from p
	 * @param kernel_q kernel to use on samples from q
	 * @param p_and_q samples from p and q, appended
	 * @param q_start index of first sample of q
	 */
	CKernelIndependenceTestStatistic(CKernel* kernel_p, CKernel* kernel_q,
			CFeatures* p_and_q, index_t q_start);

	/** Constructor.
	 * This is a convienience constructor which copies both features to one
	 * element and then calls the other constructor. Needs twice the memory
	 * for a short time
	 *
	 * @param kernel_p kernel to use on samples from p
	 * @param kernel_q kernel to use on samples from q
	 * @param p samples from distribution p, will be copied and NOT
	 * SG_REF'ed
	 * @param q samples from distribution q, will be copied and NOT
	 * SG_REF'ed
	 */
	CKernelIndependenceTestStatistic(CKernel* kernel_p, CKernel* kernel_q,
			CFeatures* p, CFeatures* q);

	/** destructor */
	virtual ~CKernelIndependenceTestStatistic();

	/** merges both sets of samples and computes the test statistic
	 * m_num_sample_iteration times. This version checks if a precomputed
	 * custom kernel is used, and, if so, just permutes it instead of re-
	 * computing it in every iteration.
	 *
	 * @return vector of all statistics
	 */
	virtual SGVector<float64_t> sample_null();

	virtual const char* get_name() const=0;

private:
	void init();

protected:
	/** underlying kernel for p */
	CKernel* m_kernel_p;

	/** underlying kernel for q */
	CKernel* m_kernel_q;
};

}

#endif /* __KERNELINDEPENDENCESTSTATISTIC_H_ */
