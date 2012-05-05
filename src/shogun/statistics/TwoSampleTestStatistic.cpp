/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/TwoSampleTestStatistic.h>
#include <shogun/features/Features.h>

using namespace shogun;

CTwoSampleTestStatistic::CTwoSampleTestStatistic() : CTestStatistic()
{
	init();
}

CTwoSampleTestStatistic::CTwoSampleTestStatistic(CFeatures* p_and_q,
		index_t q_start) :
		CTestStatistic()
{
	init();

	m_p_and_q=p_and_q;
	SG_REF(m_p_and_q);

	m_q_start=q_start;
}

CTwoSampleTestStatistic::~CTwoSampleTestStatistic()
{
	SG_UNREF(m_p_and_q);
}

void CTwoSampleTestStatistic::init()
{
	/* TODO register parameters */
	m_p_and_q=NULL;
	m_q_start=0;
}
