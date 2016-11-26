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
