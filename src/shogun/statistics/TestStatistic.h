/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __TESTSTATISTIC_H_
#define __TESTSTATISTIC_H_

#include <shogun/base/SGObject.h>

namespace shogun
{

/** @brief Test statistic base class. Provides an interface for statistical
 * tests via two methods: compute_statistic() and compute_p_value(). The second
 * computes a p-value for the statistic computed by the first method.
 * The p-value represents the position of the statistic in the null-distribution,
 * i.e. the distribution of the statistic population given the null-hypothesis
 * is true. (1-position = p-value).
 *
 * Method perform_test performs the underlying test for a given niveau alpha.
 *
 * Abstract base class.
 */
class CTestStatistic : public CSGObject
{
	public:
		CTestStatistic() {};

		virtual ~CTestStatistic() {};

		virtual float64_t compute_statistic()
		{
			SG_ERROR("%s::compute_statistic() is not implemented!\n");
			return 0.0;
		}

		virtual float64_t compute_p_value(float64_t statistic)
		{
			SG_ERROR("%s::compute_p_value() is not implemented!\n");
			return 0.0;
		}

		/** Performs a two-sample test with the current settings on the current
		 * data. Computes test statistic and compares its p-value against the
		 * desired one and returns true if the p-value is at least as good.
		 *
		 * @param alpha test niveau alpha
		 * @return true if null-hypothesis (p==q) is rejected, false otherwise
		 */
		virtual bool perform_test(float64_t alpha)
		{
			/* compute p-value and test statistic */
			float64_t statistic=compute_statistic();
			float64_t p_value=compute_p_value(statistic);

			/* null hypothesis is rejected if computed p-value is smaller than
			 * alpha */
			return p_value<alpha;
		}

		inline virtual const char* get_name() const=0;
};

}

#endif /* __TESTSTATISTIC_H_ */
