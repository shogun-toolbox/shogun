/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2014 - 2016 Soumyajit De
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

#ifndef TWO_DISTRIBUTION_TEST_H_
#define TWO_DISTRIBUTION_TEST_H_

#include <shogun/statistical_testing/HypothesisTest.h>

namespace shogun
{

class CDistance;
class CCustomDistance;

/**
 * @brief Class TwoDistributionTest is the base class for the statistical
 * hypothesis testing with samples from two distributions, \f$mathbf{P}\f$
 * and \f$\mathbf{Q}\f$.
 *
 * \sa {CTwoSampleTest}
 */
class CTwoDistributionTest : public CHypothesisTest
{
public:
	/** Default constructor */
	CTwoDistributionTest();

	/** Destrutor */
	virtual ~CTwoDistributionTest();

	/**
	 * Method that initializes the samples from \f$\mathbf{P}\f$. This method
	 * is kept virtual for the sub-classes to perform additional initialization
	 * tasks that have to be performed every time features are set/updated.
	 *
	 * @param samples_from_p The CFeatures instance representing the samples
	 * from \f$\mathbf{P}\f$.
	 */
	virtual void set_p(CFeatures* samples_from_p);

	/** @return The samples from \f$\mathbf{P}\f$. */
	CFeatures* get_p() const;

	/**
	 * Method that initializes the samples from \f$\mathbf{Q}\f$. This method
	 * is kept virtual for the sub-classes to perform additional initialization
	 * tasks that have to be performed every time features are set/updated.
	 *
	 * @param samples_from_q The CFeatures instance representing the samples
	 * from \f$\mathbf{Q}\f$.
	 */
	virtual void set_q(CFeatures* samples_from_q);

	/** @return The samples from \f$\mathbf{Q}\f$. */
	CFeatures* get_q() const;

	/**
	 * Method that initializes the number of samples to be drawn from distribution
	 * \f$\mathbf{P}\f$. Please ensure to call this method if you are intending to
	 * use streaming data generators that generate the samples on the fly. For
	 * other types of features, the number of samples is set internally from the
	 * features object itself, therefore this method should not be used.
	 *
	 * @param num_samples_from_p The CFeatures instance representing the samples
	 * from \f$\mathbf{P}\f$.
	 */
	void set_num_samples_p(index_t num_samples_from_p);

	/** @return The number of samples from \f$\mathbf{P}\f$. */
	const index_t get_num_samples_p() const;

	/**
	 * Method that initializes the number of samples to be drawn from distribution
	 * \f$\mathbf{Q}\f$. Please ensure to call this method if you are intending to
	 * use streaming data generators that generate the samples on the fly. For
	 * other types of features, the number of samples is set internally from the
	 * features object itself, therefore this method should not be used.
	 *
	 * @param num_samples_from_q The CFeatures instance representing the samples
	 * from \f$\mathbf{Q}\f$.
	 */
	void set_num_samples_q(index_t num_samples_from_q);

	/** @return The number of samples from \f$\mathbf{Q}\f$. */
	const index_t get_num_samples_q() const;

	/**
	 * Method that pre-computes the pair-wise distance between the samples using
	 * the provided distance instance.
	 *
	 * @param distance The distance instance used for pre-computing the pair-wise
	 * distance.
	 * @return A newly created CCustomDistance instance representing the
	 * pre-computed pair-wise distance between the samples.
	 */
	CCustomDistance* compute_distance(CDistance* distance);

	/**
	 * Method that pre-computes the pair-wise distance between the joint samples using
	 * the provided distance instance. A temporary object appending the samples from
	 * both the distributions is created in order to perform the task.
	 *
	 * @param distance The distance instance used for pre-computing the pair-wise
	 * distance.
	 * @return A newly created CCustomDistance instance representing the
	 * pre-computed pair-wise distance between the joint samples.
	 */
	CCustomDistance* compute_joint_distance(CDistance* distance);

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
};

}
#endif // TWO_DISTRIBUTION_TEST_H_
