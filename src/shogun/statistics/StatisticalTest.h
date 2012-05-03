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
		CStatisticalTest(CTestStatistic* statistic, float64_t confidence);

		virtual ~CStatisticalTest();

		/** TODO
		 *
		 * @return true if the NULL-hypothesis is rejected */
		virtual bool perform_test();

		inline virtual const char* get_name() const { return "StatisticalTest"; }

	private:
		void init();

	protected:
		/** Confidence niveau of the test, test correct with (1-m_confidence) */
		float64_t m_confidence;

		CTestStatistic* m_statistic;
};

}

#endif /* __STATISTICALTEST_H_ */
