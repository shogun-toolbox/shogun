/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#ifndef __TwoDistributionsTestStatistic_H_
#define __TwoDistributionsTestStatistic_H_

#include <shogun/statistics/TestStatistic.h>

namespace shogun
{

class CFeatures;

/** @brief Provides an interface for performing
 * statistical tests on two sets of samples from two distributions.
 * Instances of these tests are the classical two-sample test and the
 * independence test. This class may be used as base class for both.
 *
 * Abstract base class. Provides all interfaces and implements approximating
 * the null distribution via bootstrapping, i.e. repeatedly merging both samples
 * and them compute the test statistic on them.
 *
 */
class CTwoDistributionsTestStatistic : public CTestStatistic
{
	public:
		CTwoDistributionsTestStatistic();

		/** Constructor
		 *
		 * @param p_and_q feature data. Is assumed to contain samples from both
		 * p and q. First all samples from p, then from index q_start all
		 * samples from q
		 *
		 * @param p_and_q samples from p and q, appended
		 * @param m index of first sample of q
		 */
		CTwoDistributionsTestStatistic(CFeatures* p_and_q, index_t m);

		/** Constructor.
		 * This is a convienience constructor which copies both features to one
		 * element and then calls the other constructor. Needs twice the memory
		 * for a short time
		 *
		 * @param p samples from distribution p, will be copied and NOT
		 * SG_REF'ed
		 * @param q samples from distribution q, will be copied and NOT
		 * SG_REF'ed
		 */
		CTwoDistributionsTestStatistic(CFeatures* p, CFeatures* q);

		virtual ~CTwoDistributionsTestStatistic();

		/** merges both sets of samples and computes the test statistic
		 * m_bootstrap_iteration times
		 *
		 * @return vector of all statistics
		 */
		virtual SGVector<float64_t> bootstrap_null();

		/** computes a p-value based on current method for approximating the
		 * null-distribution. The p-value is the 1-p quantile of the null-
		 * distribution where the given statistic lies in.
		 *
		 * @param statistic statistic value to compute the p-value for
		 * @return p-value parameter statistic is the (1-p) percentile of the
		 * null distribution
		 */
		virtual float64_t compute_p_value(float64_t statistic);

		/** computes a threshold based on current method for approximating the
		 * null-distribution. The threshold is the argument of the \f$1-\alpha\f$
		 * quantile of the null. \f$\alpha\f$ is provided.
		 *
		 * @param alpha \f$\alpha\f$ quantile to get the threshold for
		 * @return threshold which is the \f$1-\alpha\f$ quantile of the null
		 * distribution
		 */
		virtual float64_t compute_threshold(float64_t alpha);

		/** Setter for joint features
		 * @param p_and_q joint features from p and q to set
		 */
		virtual void set_p_and_q(CFeatures* p_and_q);

		/** Getter for joint features, SG_REF's
		 * @return joint feature object
		 */
		virtual CFeatures* get_p_and_q();

		/** @return number of to be used samples m */
		index_t get_m() { return m_m; }

		virtual const char* get_name() const=0;

	private:
		void init();

	protected:
		/** concatenated samples of the two distributions (two blocks) */
		CFeatures* m_p_and_q;

		/** defines the first index of samples of q */
		index_t m_m;
};

}

#endif /* __TwoDistributionsTestStatistic_H_ */
