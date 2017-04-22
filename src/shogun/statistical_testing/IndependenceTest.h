/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2014 - 2017 Soumyajit De
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

#ifndef INDEPENDENCE_TEST_H_
#define INDEPENDENCE_TEST_H_

#include <memory>
#include <shogun/statistical_testing/TwoDistributionTest.h>

namespace shogun
{

class CKernel;

namespace internal
{
	class KernelManager;
}

/**
 * @brief Provides an interface for performing the independence test.
 * Given samples \f$Z=\{(x_i,y_i)\}_{i=1}^m\f$ from the joint distribution
 * \f$\textbf{P}_{xy}\f$, whether the joint distribution factorize as
 * \f$\textbf{P}_{xy}=\textbf{P}_x\textbf{P}_y\f$, i.e. product of the marginals.
 * The null-hypothesis says yes, i.e. no dependence, the alternative hypothesis
 * says no.
 *
 * Abstract base class. Provides all interfaces and implements approximating
 * the null distribution via permutation, i.e. shuffling the samples from
 * one distribution repeatedly using subsets while keeping the samples from
 * the other distribution in its original order
 *
 */
class CIndependenceTest : public CTwoDistributionTest
{
public:
	/** Default constructor */
	CIndependenceTest();

	/** Destructor */
	virtual ~CIndependenceTest();

	/**
	 * Method that sets the kernel to be used for performing the test for the
	 * samples from p.
	 *
	 * @param kernel_p The kernel instance to be used for samples from p
	 */
	void set_kernel_p(CKernel* kernel_p);

	/** @return The kernel instance that is used for samples from p */
	CKernel* get_kernel_p() const;

	/**
	 * Method that sets the kernel to be used for performing the test for the
	 * samples from q.
	 *
	 * @param kernel_q The kernel instance to be used for samples from q
	 */
	void set_kernel_q(CKernel* kernel_q);

	/** @return The kernel instance that is used for samples from q */
	CKernel* get_kernel_q() const;

	/**
	 * Interface for computing the test-statistic for the hypothesis test.
	 *
	 * @return test statistic for the given data/parameters/methods
	 */
	virtual float64_t compute_statistic()=0;

	/**
	 * Interface for computing the samples under the null-hypothesis.
	 *
	 * @return vector of all statistics
	 */
	virtual SGVector<float64_t> sample_null()=0;

	/** @return The name of the class */
	virtual const char* get_name() const;
protected:
	internal::KernelManager& get_kernel_mgr();
	const internal::KernelManager& get_kernel_mgr() const;
private:
	struct Self;
	std::unique_ptr<Self> self;
};

}
#endif // INDEPENDENCE_TEST_H_
