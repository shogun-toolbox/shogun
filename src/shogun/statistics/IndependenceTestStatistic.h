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
/** @brief Independence test base class. Provides an interface for performing an
 * independence test, i.e. Given samples from two distributions p_x and p_y, the
 * null-hypothesis is: H0: p_xy=p_x*p_y, the alternative hypothesis:
 * H1: p_xy!=p_x*p_y
 *
 * Abstract base class.
 *
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
};

}

#endif /* __INDEPENDENCETESTSTATISTIC_H_ */
