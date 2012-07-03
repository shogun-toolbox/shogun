/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __INDEPENDENCETESTSTATISTIC_H_
#define __INDEPENDENCETESTSTATISTIC_H_

#include <shogun/statistics/TestStatistic.h>

namespace shogun
{

class CFeatures;
/** TODO
 *
 * @brief Test statistic base class. Provides an interface for statistical
 * tests via three methods: compute_statistic(), compute_p_value() and
 * compute_threshold(). The second computes a p-value for the statistic computed
 * by the first method.
 * The p-value represents the position of the statistic in the null-distribution,
 * i.e. the distribution of the statistic population given the null-hypothesis
 * is true. (1-position = p-value).
 * The third method,  compute_threshold(), computes a threshold for a given
 * test level which is needed to reject the null-hypothesis
 *
 * Abstract base class.
 */
class CIndependenceTestStatistic : public CTestStatistic
{
	public:
		CIndependenceTestStatistic();

		CIndependenceTestStatistic(CFeatures* p, CFeatures* q);

		virtual ~CIndependenceTestStatistic();

		/** merges both sets of samples and computes the test statistic
		 * m_bootstrap_iteration times
		 *
		 * @return vector of all statistics
		 */
		virtual SGVector<float64_t> bootstrap_null();

		/** computes a p-value based on bootstrapping the null-distribution.
		 * This method should be overridden for different methods
		 *
		 * @param statistic statistic value to compute the p-value for
		 * @return p-value parameter statistic is the (1-p) percentile of the
		 * null distribution
		 */
		virtual float64_t compute_p_value(float64_t statistic);

		inline virtual const char* get_name() const=0;

	private:
		void init();

	protected:
		/** samples from p */
		CFeatures* m_p;

		/** samples from q */
		CFeatures* m_q;

		/** number of iterations for bootstrapping null-distributions */
		index_t m_bootstrap_iterations;

		/** Defines how the the null distribution is approximated */
		ENullApproximationMethod m_null_approximation_method;
};

}

#endif /* __INDEPENDENCETESTSTATISTIC_H_ */
