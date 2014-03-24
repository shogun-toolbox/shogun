/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012-2013 Heiko Strathmann
 * Written (w) 2014 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef KERNEL_INDEPENDENCE_TEST_H_
#define KERNEL_INDEPENDENCE_TEST_H_

#include <shogun/lib/config.h>
#include <shogun/statistics/IndependenceTest.h>

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
 * CIndependenceTest::sample_null() otherwise, which requires to re-compute
 * the kernel in each iteration via subsets on the features instead
 *
 * Abstract base class.
 */
class CKernelIndependenceTest: public CIndependenceTest
{
public:
	/** default constructor */
	CKernelIndependenceTest();

	/** Constructor.
	 *
	 * Initializes the kernels and features from the two distributions and
	 * SG_REFs them
	 *
	 * @param kernel_p kernel to use on samples from p
	 * @param kernel_q kernel to use on samples from q
	 * @param p samples from distribution p
	 * @param q samples from distribution q
	 */
	CKernelIndependenceTest(CKernel* kernel_p, CKernel* kernel_q,
			CFeatures* p, CFeatures* q);

	/** destructor */
	virtual ~CKernelIndependenceTest();

	/** shuffles the indeices that corresponds to the kernel entries of
	 * samples from p while accessing samples from q in the original order and
	 * computes the test statistic  m_num_null_samples times. This version
	 * checks if a precomputed custom kernel is used, and, if so, just permutes
	 * the indices of the kernel corresponding to p instead of re-computing it
	 * in every iteration.
	 *
	 * @return vector of all statistics
	 */
	virtual SGVector<float64_t> sample_null();

	/** @return the class name */
	virtual const char* get_name() const=0;

private:
	/** register parameters and intiailize with default values */
	void init();

protected:
	/** underlying kernel for p */
	CKernel* m_kernel_p;

	/** underlying kernel for q */
	CKernel* m_kernel_q;
};

}

#endif /* KERNEL_INDEPENDENCE_TEST_H_ */
