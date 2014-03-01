/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 * Written (W) 2014 Soumyajit De
 */

#include <shogun/statistics/IndependenceTestStatistic.h>
#include <shogun/features/Features.h>

using namespace shogun;

CIndependenceTestStatistic::CIndependenceTestStatistic() :
		CTestStatistic()
{
	init();
}

CIndependenceTestStatistic::CIndependenceTestStatistic(
		CFeatures* p_and_q,
		index_t m) : CTestStatistic()
{
	init();

	m_p_and_q=p_and_q;
	SG_REF(m_p_and_q);

	m_m=m;
}

CIndependenceTestStatistic::CIndependenceTestStatistic(
		CFeatures* p, CFeatures* q) :
		CTestStatistic()
{
	init();

	m_p_and_q=p->create_merged_copy(q);
	SG_REF(m_p_and_q);

	m_m=p->get_num_vectors();
}

CIndependenceTestStatistic::~CIndependenceTestStatistic()
{
	SG_UNREF(m_p_and_q);
}

void CIndependenceTestStatistic::init()
{
	SG_ADD((CSGObject**)&m_p_and_q, "p_and_q", "Concatenated samples p and q",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_m, "m", "Index of first sample of q",
			MS_NOT_AVAILABLE);

	m_p_and_q=NULL;
	m_m=0;
}

SGVector<float64_t> CIndependenceTestStatistic::sample_null()
{
	SG_DEBUG("entering CIndependenceTestStatistic::sample_null()\n")

	REQUIRE(m_p_and_q, "CIndependenceTestStatistic::sample_null(): "
			"No appended features p and q!\n");

	/* compute sample statistics for null distribution */
	SGVector<float64_t> results(m_num_permutation_iterations);

	/* memory for index permutations. Adding of subset has to happen
	 * inside the loop since it may be copied if there already is one set.
	 *
	 * subset for selecting samples from p and q. In this case we want to
	 * shuffle only samples from p while keeping samples from q fixed. But
	 * since its a subset stack, we need all the indices to be available.
	 * We will however permute the samples from p only (first m_m samples) */
	SGVector<index_t> ind_permutation(2*m_m);
	ind_permutation.range_fill();

	for (index_t i=0; i<m_num_permutation_iterations; ++i)
	{
		/* idea: merge features of p and q, shuffle samples from p while
		 * keeping samples from q fixed and compute statistic.
		 * This is done using subsets here */

		/* create index permutation and add as subset. we only need to permute
		 * the first m_m indices which will shuffle the samples from p */
		SGVector<index_t>::permute(ind_permutation.vector, m_m);

		/* compute statistic for this permutation of mixed samples */
		m_p_and_q->add_subset(ind_permutation);
		results[i]=compute_statistic();
		m_p_and_q->remove_subset();
	}

	SG_DEBUG("leaving CIndependenceTestStatistic::sample_null()\n")
	return results;
}

float64_t CIndependenceTestStatistic::compute_p_value(
		float64_t statistic)
{
	float64_t result=0;

	if (m_null_approximation_method==PERMUTATION)
	{
		/* sample a bunch of MMD values from null distribution */
		SGVector<float64_t> values=sample_null();

		/* find out percentile of parameter "statistic" in null distribution */
		values.qsort();
		float64_t i=values.find_position_to_insert(statistic);

		/* return corresponding p-value */
		result=1.0-i/values.vlen;
	}
	else
	{
		SG_ERROR("CIndependenceTestStatistics::compute_p_value(): Unknown"
				" method to approximate null distribution!\n");
	}

	return result;
}

float64_t CIndependenceTestStatistic::compute_threshold(
		float64_t alpha)
{
	float64_t result=0;

	if (m_null_approximation_method==PERMUTATION)
	{
		/* sample a bunch of MMD values from null distribution */
		SGVector<float64_t> values=sample_null();

		/* return value of (1-alpha) quantile */
		result=values[index_t(CMath::floor(values.vlen*(1-alpha)))];
	}
	else
	{
		SG_ERROR("CIndependenceTestStatistics::compute_threshold():"
				"Unknown method to approximate null distribution!\n");
	}

	return result;
}

void CIndependenceTestStatistic::set_p_and_q(CFeatures* p_and_q)
{
	/* ref before unref to avoid problems when instances are equal */
	SG_REF(p_and_q);
	SG_UNREF(m_p_and_q);
	m_p_and_q=p_and_q;
}

CFeatures* CIndependenceTestStatistic::get_p_and_q()
{
	SG_REF(m_p_and_q);
	return m_p_and_q;
}

