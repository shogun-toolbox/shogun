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

		inline virtual const char* get_name() const { return "StatisticalTest"; }

	private:
		void init();

	protected:
		CTestStatistic* m_statistic;
};

}

#endif /* __STATISTICALTEST_H_ */
