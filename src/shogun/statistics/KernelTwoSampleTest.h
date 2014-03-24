/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#ifndef KERNEL_TWO_SAMPLE_TEST_H_
#define KERNEL_TWO_SAMPLE_TEST_H_

#include <shogun/lib/config.h>
#include <shogun/statistics/TwoSampleTest.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{

class CFeatures;
class CKernel;

/** @brief Kernel two sample test base class. Provides an interface for
 * performing a two-sample test using a kernel, i.e. Given samples from two
 * distributions \f$p\f$ and \f$q\f$, the null-hypothesis is: \f$H_0: p=q\f$,
 * the alternative hypothesis: \f$H_1: p\neq q\f$.
 *
 * In this class, this is done using a single kernel for the data.
 *
 * The class also re-implements the sample_null() method. If the underlying
 * kernel is a custom one (precomputed), the rows and column of the kernel
 * matrix is permuted using subsets. The computation falls back to
 * CTwoSampleTest::sample_null() otherwise.
 *
 * Abstract base class.
 */
class CKernelTwoSampleTest : public CTwoSampleTest
{
	public:
		/** default constructor */
		CKernelTwoSampleTest();

		/** Constructor
		 *
		 * @param p_and_q feature data. Is assumed to contain samples from both
		 * p and q. First all samples from p, then from index q_start all
		 * samples from q
		 *
		 * @param kernel kernel to use
		 * @param p_and_q samples from p and q, appended
		 * @param q_start index of first sample of q
		 */
		CKernelTwoSampleTest(CKernel* kernel, CFeatures* p_and_q,
				index_t q_start);

		/** Constructor.
		 * This is a convienience constructor which copies both features to one
		 * element and then calls the other constructor. Needs twice the memory
		 * for a short time
		 *
		 * @param kernel kernel for MMD
		 * @param p samples from distribution p, will be copied and NOT
		 * SG_REF'ed
		 * @param q samples from distribution q, will be copied and NOT
		 * SG_REF'ed
		 */
		CKernelTwoSampleTest(CKernel* kernel, CFeatures* p,
				CFeatures* q);

		/** destructor */
		virtual ~CKernelTwoSampleTest();

		/** Setter for the underlying kernel
		 * @param kernel new kernel to use
		 */
		inline virtual void set_kernel(CKernel* kernel)
		{
			/* ref before unref to prevent deleting in case objects are the same */
			SG_REF(kernel);
			SG_UNREF(m_kernel);
			m_kernel=kernel;
		}

		/** @return underlying kernel, is SG_REF'ed */
		inline virtual CKernel* get_kernel()
		{
			SG_REF(m_kernel);
			return m_kernel;
		}

		/** merges both sets of samples and computes the test statistic
		 * m_num_null_samples times. This version checks if a precomputed
		 * custom kernel is used, and, if so, just permutes it instead of re-
		 * computing it in every iteration.
		 *
		 * @return vector of all statistics
		 */
		virtual SGVector<float64_t> sample_null();

		/** Same as compute_statistic(), but with the possibility to perform on
		 * multiple kernels at once
		 *
		 * @param multiple_kernels if true, and underlying kernel is K_COMBINED,
		 * method will be executed on all subkernels on the same data
		 * @return vector of results for subkernels
		 */
		virtual SGVector<float64_t> compute_statistic(
				bool multiple_kernels)=0;

		/** Wrapper for compute_statistic(false) */
		virtual float64_t compute_statistic()=0;

		virtual const char* get_name() const=0;

	private:
		void init();

	protected:
		/** underlying kernel */
		CKernel* m_kernel;
};

}

#endif /* KERNEL_TWO_SAMPLE_TEST_H_ */
