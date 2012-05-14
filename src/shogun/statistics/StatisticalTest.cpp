/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/StatisticalTest.h>
#include <shogun/statistics/TestStatistic.h>

using namespace shogun;

CStatisticalTest::CStatisticalTest() : CSGObject()
{
	init();
}

CStatisticalTest::CStatisticalTest(CTestStatistic* statistic) : CSGObject()
{
	init();

	m_statistic=statistic;
	SG_REF(m_statistic);
}

CStatisticalTest::~CStatisticalTest()
{
	SG_UNREF(m_statistic);
}

float64_t CStatisticalTest::perform_test()
{
	if (!m_statistic)
	{
		SG_ERROR("CStatisticalTest::perform_test(): No object to compute test "
				"statistic!\n");
	}

	float64_t statistic=m_statistic->compute_statistic();
	return m_statistic->compute_p_value(statistic);
}

void CStatisticalTest::init()
{
	/* TODO register parameters*/

	m_statistic=NULL;
}
