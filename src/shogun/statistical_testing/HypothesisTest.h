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

#ifndef HYPOTHESIS_TEST_H_
#define HYPOTHESIS_TEST_H_

#include <memory>
#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>

namespace shogun
{

class CFeatures;

namespace internal
{

class DataManager;

}

/**
 * @brief Hypothesis test base class. Provides an interface for statistical
 * hypothesis testing via three methods: compute_statistic(), compute_p_value()
 * and compute_threshold(). The second computes a p-value for the statistic
 * computed by the first method. The p-value represents the position of the
 * statistic in the null-distribution, i.e. the distribution of the statistic
 * population given the null-hypothesis is true. (1-position = p-value).
 *
 * The third method,  compute_threshold(), computes a threshold for a given
 * test level which is needed to reject the null-hypothesis.
 *
 * Also provides an interface for sampling from the null-distribution.
 * The actual sampling has to be implemented in sub-classes.
 *
 * All statistical tests should inherit from this class.
 *
 * Abstract base class.
 */
class CHypothesisTest : public CSGObject
{
public:
	/** Default constructor */
	CHypothesisTest();

	/** Destructor */
	virtual ~CHypothesisTest();

	/**
	 * Method that enables/disables the training-testing mode. If this option
	 * is turned on, then the samples would be split in two pieces: one chunk
	 * would be used for training algorithms and the other chunk would be used
	 * for performing tests. If this option is turned off, the entire data
	 * would be used for performing the test. Before running any training
	 * algorithms, make sure to turn this mode on.
	 *
	 * By default, the training-testing mode is turned off.
	 *
	 * \sa {set_train_test_ratio()}
	 *
	 * @param on Whether to enable/disable the training-testing mode
	 */
	void set_train_test_mode(bool on);

	/**
	 * Method that specifies the ratio of training-testing data split for the
	 * algorithms. Note that this is NOT the percentage of samples to be used
	 * for training, rather the ratio of the number of samples to be used for
	 * training and that of testing.
	 *
	 * By default, an equal 50-50 split (ratio = 1) is made.
	 *
	 * \sa {set_train_test_mode()}
	 *
	 * @param ratio The ratio of the number of samples to be used for training
	 * and that of testing
	 */
	void set_train_test_ratio(float64_t ratio);

	/**
	 * Method that computes a p-value based on current method for approximating
	 * the null-distribution. The p-value is the 1-p quantile of the null-
	 * distribution where the given statistic lies in.
	 *
	 * This method depends on the implementation of sample_null method
	 * which should be implemented by the sub-classes.
	 *
	 * @param statistic statistic value to compute the p-value for
	 * @return p-value parameter statistic is the (1-p) percentile of the
	 * null distribution
	 */
	virtual float64_t compute_p_value(float64_t statistic);

	/**
	 * Method that computes a threshold based on current method for approximating
	 * the null-distribution. The threshold is the value that a statistic has
	 * to have in ordner to reject the null-hypothesis.
	 *
	 * This method depends on the implementation of sample_null method
	 * which should be implemented by the sub-classes.
	 *
	 * @param alpha test level to reject null-hypothesis
	 * @return Threshold for statistics to reject null-hypothesis
	 */
	virtual float64_t compute_threshold(float64_t alpha);

	/**
	 * Method that performs the complete hypothesis test on current data and
	 * returns a binary answer: wheter null hypothesis is rejected or not.
	 *
	 * This is just a wrapper for the above compute_p_value() method that
	 * returns a p-value. If this p-value lies below the test level alpha,
	 * the null hypothesis is rejected.
	 *
	 * Should not be overwritten in subclasses. (Therefore not virtual)
	 *
	 * @param alpha test level alpha.
	 * @return true if null hypothesis is rejected and false otherwise
	 */
	bool perform_test(float64_t alpha);

	/**
	 * Interface for computing the test-statistic for the hypothesis test.
	 *
	 * @return Test statistic for the given data/parameters/methods
	 */
	virtual float64_t compute_statistic()=0;

	/**
	 * Interface for computing the samples under the null-hypothesis.
	 *
	 * @return Vector of all statistics
	 */
	virtual SGVector<float64_t> sample_null()=0;

	/** @return The name of the class */
	virtual const char* get_name() const;
protected:
	explicit CHypothesisTest(index_t num_distributions);
	internal::DataManager& get_data_mgr();
	const internal::DataManager& get_data_mgr() const;
private:
	CHypothesisTest(const CHypothesisTest& other)=delete;
	CHypothesisTest& operator=(const CHypothesisTest& other)=delete;

	struct Self;
	std::unique_ptr<Self> self;
};

}

#endif // HYPOTHESIS_TEST_H_
