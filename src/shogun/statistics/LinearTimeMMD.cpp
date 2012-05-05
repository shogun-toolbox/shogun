/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/features/Features.h>

using namespace shogun;

CLinearTimeMMD::CLinearTimeMMD() : CTwoSampleTestStatistic()
{
	init();
}

CLinearTimeMMD::CLinearTimeMMD(CKernel* kernel, CFeatures* p_and_q,
		index_t q_start) :CTwoSampleTestStatistic(p_and_q, q_start)
{
	init();

	if (q_start!=p_and_q->get_num_vectors()/2)
	{
		SG_ERROR("CLinearTimeMMD: Only features with equal number of vectors "
				"are currently possible\n");
	}

	m_kernel=kernel;
	SG_REF(kernel);
}

CLinearTimeMMD::~CLinearTimeMMD()
{
	SG_UNREF(m_kernel);
}

void CLinearTimeMMD::init()
{
	/* TODO register parameters*/

	m_kernel=NULL;
	m_threshold_method=MMD_BOOTSTRAP;
	m_bootstrap_iterations=100;
}

float64_t CLinearTimeMMD::compute_statistic()
{
	/* TODO maybe add parallelized kernel matrix trace method to CKernel? */
	/* TODO features with a different number of vectors should be allowed */

	/* m is number of samples from each distribution, m_2 is half of it
	 * using names from JLMR paper (see class documentation) */
	index_t m=m_q_start;
	index_t m_2=m/2;

	/* allocate memory */
	SGVector<float64_t> tr_K_x(m_2);
	SGVector<float64_t> tr_K_y(m_2);
	SGVector<float64_t> tr_K_xy(m);

	/* compute traces of kernel matrices for linear MMD */
	m_kernel->init(m_p_and_q, m_p_and_q);

	/* p and p */
	for (index_t i=0; i<m_2; ++i)
		tr_K_x.vector[i]=m_kernel->kernel(i, m_2+i);

	/* q and q */
	for (index_t i=m_q_start; i<m+m_2; ++i)
		tr_K_y.vector[i-m_q_start]=m_kernel->kernel(i, m_2+i);

	/* p and q */
	for (index_t i=0; i<m; ++i)
		tr_K_xy.vector[i]=m_kernel->kernel(i, m+i);

	/* compute result */
	float64_t first=0;
	float64_t second=0;
	float64_t third=0;

	for (index_t i=0; i<m_2; ++i)
	{
		first+=tr_K_x.vector[i];
		second+=tr_K_y.vector[i];
		third+=tr_K_xy.vector[i];
	}

	for (index_t i=m_2; i<m; ++i)
		third+=tr_K_xy.vector[i-m_2];

	return 1.0/m_2*(first+second)+1.0/m*third;
}

float64_t CLinearTimeMMD::compute_threshold(float64_t confidence)
{
	float64_t result=0;

	switch (m_threshold_method)
	{
	case MMD_BOOTSTRAP:
		result=bootstrap_threshold(confidence);
		break;

	default:
		SG_ERROR("%s::compute_threshold(): Unknown method to compute"
				" threshold!\n");

	}

	return result;
}

float64_t CLinearTimeMMD::bootstrap_threshold(float64_t confidence)
{
	/* compute mean of all bootstrap statistics using running averages */
	SGVector<float64_t> results(m_bootstrap_iterations);

	/* memory for index permutations, (would slow down loop) */
	SGVector<index_t> ind_permutation(m_p_and_q->get_num_vectors());
	ind_permutation.range_fill();

	for (index_t i=0; i<m_bootstrap_iterations; ++i)
	{
		/* idea: merge features of p and q, shuffle, and compute statistic.
		 * This is done using subsets here */

		/* create index permutation and add as subset. This will mix samples
		 * from p and q */
		CMath::permute_vector(ind_permutation);
		m_p_and_q->add_subset(ind_permutation);

		/* compute statistic for this permutation of mixed samples */
		results.vector[i]=compute_statistic();

		/* clean up */
		m_p_and_q->remove_subset();
	}

	/* compute threshold, sort elements and return the one that corresponds to
	 * confidence niveau */
	CMath::qsort(results.vector, results.vlen);
	index_t result_idx=CMath::round((1-confidence)*results.vlen);
	float64_t result=results.vector[result_idx];

	/* clean up and return */
	return result;
}
