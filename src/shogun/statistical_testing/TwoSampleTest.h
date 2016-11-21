/*
 * Restructuring Shogun's statistical hypothesis testing framework.
 * Copyright (C) 2016  Soumyajit De
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
