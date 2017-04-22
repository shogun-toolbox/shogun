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

#ifndef TWO_SAMPLE_TEST_H_
#define TWO_SAMPLE_TEST_H_

#include <memory>
#include <shogun/statistical_testing/TwoDistributionTest.h>

namespace shogun
{

class CKernel;
class CFeatures;

namespace internal
{
	class KernelManager;
}

/** @brief Kernel two sample test base class. Provides an interface for
 * performing a two-sample test using a kernel, i.e. Given samples from two
 * distributions \f$p\f$ and \f$q\f$, the null-hypothesis is: \f$H_0: p=q\f$,
 * the alternative hypothesis: \f$H_1: p\neq q\f$.
 *
 * In this class, this is done using a single kernel for the data.
 *
 * Abstract base class.
 */
class CTwoSampleTest : public CTwoDistributionTest
{
public:
	/** Default constructor */
	CTwoSampleTest();

	/**
	 * Convenience constructor that initializes the samples from two distributions.
	 *
	 * @param samples_from_p Samples from \f$p\f$
	 * @param samples_from_q Samples from \f$q\f$
	 */
	CTwoSampleTest(CFeatures* samples_from_p, CFeatures* samples_from_q);

	/** Destructor */
	virtual ~CTwoSampleTest();

	/**
	 * Method that sets the kernel that is used for performing the two-sample test.
	 * It is kept virtual so that sub-classes can perform other initialization tasks
	 * that has to be trigger every time a kernel is set/updated.
	 *
	 * @param kernel The kernel instance.
	 */
	virtual void set_kernel(CKernel* kernel);

	/** @return The kernel instance that is presently being used for performing the test */
	CKernel* get_kernel() const;

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
#endif // TWO_SAMPLE_TEST_H_
