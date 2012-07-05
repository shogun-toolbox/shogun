/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/IndependenceTestStatistic.h>
#include <shogun/features/Features.h>

using namespace shogun;

CIndependenceTestStatistic::CIndependenceTestStatistic() :
		CTestStatistic()
{
	init();
}

CIndependenceTestStatistic::CIndependenceTestStatistic(CFeatures* p,
		CFeatures* q) : CTestStatistic()
{
	init();

	m_p=p;
	SG_REF(m_p);

	m_q=q;
	SG_REF(m_q);
}

CIndependenceTestStatistic::~CIndependenceTestStatistic()
{
	SG_UNREF(m_p);
	SG_UNREF(m_q);
}

void CIndependenceTestStatistic::init()
{
	SG_ADD((CSGObject**)&m_p, "p", "Samples from p", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_q, "q", "Samples from q", MS_NOT_AVAILABLE);

	m_p=NULL;
	m_q=NULL;
}

SGVector<float64_t> CIndependenceTestStatistic::bootstrap_null()
{
	/* compute bootstrap statistics for null distribution */
	SGVector<float64_t> results(m_bootstrap_iterations);

	/* clean up and return */
	return results;
}

float64_t CIndependenceTestStatistic::compute_p_value(float64_t statistic)
{
	float64_t result=0;

	if (m_null_approximation_method==BOOTSTRAP)
	{
		/* bootstrap a bunch of MMD values from null distribution */
		SGVector<float64_t> values=bootstrap_null();

		/* find out percentile of parameter "statistic" in null distribution */
		CMath::qsort(values);
		float64_t i=CMath::find_position_to_insert(values, statistic);

		/* return corresponding p-value */
		result=1.0-i/values.vlen;
	}
	else
	{
		SG_ERROR("%s::compute_p_value(): Unknown method to compute"
				" p-value!\n");
	}

	return result;
}

