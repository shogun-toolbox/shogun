/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __STATISTICALTEST_H_
#define __STATISTICALTEST_H_

#include <shogun/base/SGObject.h>

namespace shogun
{

class CTestStatistic;

/** @brief Class for statistical hypothesis tests.
 *
 * Given a test statistic, this class provides interfaces for performing
 * statistical tests which includes computing p-values as well as boolean
 * results.
 */
class CStatisticalTest : public CSGObject
{
	public:
		CStatisticalTest();
		CStatisticalTest(CTestStatistic* statistic);

		virtual ~CStatisticalTest();

		/** Performs the underlying statistical test. Returns p-value, which
		 * corresponds to the (1-p) percentile of the test's resulting statistic
		 * in the null distribution.
		 *
		 * @return p-value of test result */
		virtual float64_t perform_test();

		/** Performs a test with the current statistic and settings on current
		 * data. Computes test statistic and compares its p-value against the
		 * desired one and returns true if the p-value is at least as good.
		 *
		 * @param alpha test niveau alpha
		 * @return true if null-hypothesis (p==q) is rejected, false otherwise
		 */
		virtual bool perform_binary_test(float64_t alpha);

		/** sets a new test statistic, replacing the old one */
		void set_statistic(CTestStatistic* statistic);

		inline virtual const char* get_name() const { return "StatisticalTest"; }

	private:
		void init();

	protected:
		CTestStatistic* m_statistic;
};

}

#endif /* __STATISTICALTEST_H_ */
