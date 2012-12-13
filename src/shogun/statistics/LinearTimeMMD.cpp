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
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/kernel/CombinedKernel.h>

#include <shogun/lib/external/libqp.h>

using namespace shogun;

CLinearTimeMMD::CLinearTimeMMD() :
		CKernelTwoSampleTestStatistic()
{
	init();
}

CLinearTimeMMD::CLinearTimeMMD(CKernel* kernel, CStreamingFeatures* p,
		CStreamingFeatures* q, index_t m, index_t blocksize) :
		CKernelTwoSampleTestStatistic(kernel, NULL, m)
{
	init();

	m_streaming_p=p;
	SG_REF(m_streaming_p);

	m_streaming_q=q;
	SG_REF(m_streaming_q);

	m_blocksize=blocksize;
}

CLinearTimeMMD::~CLinearTimeMMD()
{
	SG_UNREF(m_streaming_p);
	SG_UNREF(m_streaming_q);

	/* m_kernel is SG_UNREFed in base desctructor */
}

void CLinearTimeMMD::init()
{
	m_streaming_p=NULL;
	m_streaming_q=NULL;
	m_blocksize=10000;

	SG_WARNING("%s::init(): register params!\n", get_name());
}

void CLinearTimeMMD::compute_statistic_and_variance(
		float64_t& statistic, float64_t& variance)
{
	SG_DEBUG("entering %s::compute_statistic_and_variance()\n", get_name());

	REQUIRE(m_streaming_p, "%s::compute_statistic_and_variance: streaming "
			"features p required!\n", get_name());
	REQUIRE(m_streaming_q, "%s::compute_statistic_and_variance: streaming "
			"features q required!\n", get_name());

	REQUIRE(m_kernel, "%s::compute_statistic_and_variance: kernel needed!\n",
			get_name());

	/* m is number of samples from each distribution, m_2 is half of it
	 * using names from JLMR paper (see class documentation) */
	index_t m_2=m_m/2;

	SG_DEBUG("m_m=%d\n", m_m);

	/* these sums are needed to compute online statistic/variance */
	float64_t mean=0;
	float64_t M2=0;

	/* temp variable in the algorithm */
	float64_t current;
	float64_t delta;

	index_t num_examples_processed=0;
	index_t term_counter=1;
	while (num_examples_processed<m_2)
	{
		/* number of example to look at in this iteration */
		index_t num_this_run=CMath::min(m_blocksize, m_2-num_examples_processed);
		SG_DEBUG("processing %d more examples. %d so far processed. Blocksize "
				"is %d\n", num_this_run, num_examples_processed, m_blocksize);

		/* stream data from both distributions */
		CFeatures* p1=m_streaming_p->get_streamed_features(num_this_run);
		CFeatures* p2=m_streaming_p->get_streamed_features(num_this_run);
		CFeatures* q1=m_streaming_q->get_streamed_features(num_this_run);
		CFeatures* q2=m_streaming_q->get_streamed_features(num_this_run);
		SG_REF(p1);
		SG_REF(p2);
		SG_REF(q1);
		SG_REF(q2);

		/* compute kernel matrix diagonals */
		SG_DEBUG("processing kernel diagonal pp\n");
		m_kernel->init(p1, p2);
		SGVector<float64_t> pp=m_kernel->get_kernel_diagonal();

		SG_DEBUG("processing kernel diagonal qq\n");
		m_kernel->init(q1, q2);
		SGVector<float64_t> qq=m_kernel->get_kernel_diagonal();

		SG_DEBUG("processing kernel diagonal pq\n");
		m_kernel->init(p1, q2);
		SGVector<float64_t> pq=m_kernel->get_kernel_diagonal();

		SG_DEBUG("processing kernel diagonal qp\n");
		m_kernel->init(q1, p2);
		SGVector<float64_t> qp=m_kernel->get_kernel_diagonal();

		/* update mean and variance using Knuth's online variance algorithm.
		 * C.f. for example Wikipedia */
		for (index_t i=0; i<num_this_run; ++i)
		{
			/* compute sum of current h terms */
			current=pp[i]+qq[i]-pq[i]-qp[i];

			/* D. Knuth's online variance algorithm */
			delta=current-mean;
			mean=mean+delta/term_counter++;
			M2=M2+delta*(current-mean);

			SG_DEBUG("burst: current=%f, delta=%f, mean=%f, M2=%f\n",
					current, delta, mean, M2);
		}

		/* clean up */
		SG_UNREF(p1);
		SG_UNREF(p2);
		SG_UNREF(q1);
		SG_UNREF(q2);

		/* add number of processed examples for this run */
		num_examples_processed+=num_this_run;
	}
	SG_DEBUG("Done compouting statistic, processed 2*%d examples.\n",
			num_examples_processed);

	/* mean of sum all traces is linear time mmd */
	statistic=mean;
	SG_DEBUG("statistic %f\n", statistic);

	/* variance of terms can be computed using mean (statistic).
	 * Note that the variance needs to be divided by m_2 in order to get
	 * variance of null-distribution */
	variance=M2/(m_2-1)/m_2;
	SG_DEBUG("variance: %f\n", variance);

	SG_DEBUG("leaving %s::compute_statistic_and_variance()\n", get_name());
}

SGVector<float64_t> CLinearTimeMMD::compute_h_terms()
{
	SG_DEBUG("entering %s::compute_h_terms()\n", get_name());

	REQUIRE(m_streaming_p, "%s::compute_h_terms: streaming "
			"features p required!\n", get_name());
	REQUIRE(m_streaming_q, "%s::compute_h_terms: streaming "
			"features q required!\n", get_name());

	REQUIRE(m_kernel, "%s::compute_h_terms: kernel needed!\n",
			get_name());

	/* the method is basically the same as compute_variance_and_statistic(),
	 * however, it does not sum up but rather store all terms for the MMD */

	/* m is number of samples from each distribution, m_2 is half of it
	 * using names from JLMR paper (see class documentation) */
	index_t m_2=m_m/2;
	SG_DEBUG("m_m=%d\n", m_m);

	/* allocate space for result */
	SGVector<float64_t> h(m_2);

	/* these sums are needed to compute online statistic/variance */
	index_t num_examples_processed=0;
	while (num_examples_processed<m_2)
	{
		/* number of example to look at in this iteration */
		index_t num_this_run=CMath::min(m_blocksize, m_2-num_examples_processed);
		SG_DEBUG("processing %d more examples. %d so far processed. Blocksize "
				"is %d\n", num_this_run, num_examples_processed, m_blocksize);

		/* stream data from both distributions */
		CFeatures* p1=m_streaming_p->get_streamed_features(num_this_run);
		CFeatures* p2=m_streaming_p->get_streamed_features(num_this_run);
		CFeatures* q1=m_streaming_q->get_streamed_features(num_this_run);
		CFeatures* q2=m_streaming_q->get_streamed_features(num_this_run);
		SG_REF(p1);
		SG_REF(p2);
		SG_REF(q1);
		SG_REF(q2);

		/* compute kernel matrix diagonals */
		SG_DEBUG("processing kernel diagonal pp\n");
		m_kernel->init(p1, p2);
		SGVector<float64_t> pp=m_kernel->get_kernel_diagonal();

		SG_DEBUG("processing kernel diagonal qq\n");
		m_kernel->init(q1, q2);
		SGVector<float64_t> qq=m_kernel->get_kernel_diagonal();

		SG_DEBUG("processing kernel diagonal pq\n");
		m_kernel->init(p1, q2);
		SGVector<float64_t> pq=m_kernel->get_kernel_diagonal();

		SG_DEBUG("processing kernel diagonal qp\n");
		m_kernel->init(q1, p2);
		SGVector<float64_t> qp=m_kernel->get_kernel_diagonal();

		/* fill in processed part of h-term */
		for (index_t i=0; i<pp.vlen; ++i)
			h[num_examples_processed+i]=pp[i]+qq[i]-pq[i]-qp[i];

		/* clean up */
		SG_UNREF(p1);
		SG_UNREF(p2);
		SG_UNREF(q1);
		SG_UNREF(q2);

		/* add number of processed examples for this run */
		num_examples_processed+=num_this_run;
	}
	SG_DEBUG("Done compouting h-terms, processed 2*%d examples.\n",
			num_examples_processed);

	SG_WARNING("%s::compute_h_terms(): Not yet tested!\n");

	SG_DEBUG("leaving %s::compute_h_terms()\n", get_name());
	return h;
}

float64_t CLinearTimeMMD::compute_statistic()
{
	float64_t statistic=0;
	float64_t variance=0;

	/* use wrapper method */
	compute_statistic_and_variance(statistic, variance);

	return statistic;
}

float64_t CLinearTimeMMD::compute_variance_estimate()
{
	float64_t statistic=0;
	float64_t variance=0;

	/* use wrapper method */
	compute_statistic_and_variance(statistic, variance);

	return variance;
}

float64_t CLinearTimeMMD::compute_p_value(float64_t statistic)
{
	float64_t result=0;

	switch (m_null_approximation_method)
	{
	case MMD1_GAUSSIAN:
		{
			/* compute variance and use to estimate Gaussian distribution */
			float64_t std_dev=CMath::sqrt(compute_variance_estimate());
			result=1.0-CStatistics::normal_cdf(statistic, std_dev);
		}
		break;

	default:
		/* bootstrapping is handled here */
		result=CKernelTwoSampleTestStatistic::compute_p_value(statistic);
		break;
	}

	return result;
}

float64_t CLinearTimeMMD::compute_threshold(float64_t alpha)
{
	float64_t result=0;

	switch (m_null_approximation_method)
	{
	case MMD1_GAUSSIAN:
		{
			/* compute variance and use to estimate Gaussian distribution */
			float64_t std_dev=CMath::sqrt(compute_variance_estimate());
			result=1.0-CStatistics::inverse_normal_cdf(1-alpha, 0, std_dev);
		}
		break;

	default:
		/* bootstrapping is handled here */
		result=CKernelTwoSampleTestStatistic::compute_threshold(alpha);
		break;
	}

	return result;
}

float64_t CLinearTimeMMD::perform_test()
{
	float64_t result=0;

	switch (m_null_approximation_method)
	{
	case MMD1_GAUSSIAN:
		{
			/* compute statistic and variance in the same loop */
			float64_t statistic;
			float64_t variance;
			compute_statistic_and_variance(statistic, variance);

			/* estimate Gaussian distribution */
			result=1.0-CStatistics::normal_cdf(statistic,
					CMath::sqrt(variance));
		}
		break;

	default:
		/* bootstrapping can be done separately in superclass */
		result=CTestStatistic::perform_test();
		break;
	}

	return result;
}

SGVector<float64_t> CLinearTimeMMD::bootstrap_null()
{
	SGVector<float64_t> samples(m_bootstrap_iterations);

	/* instead of permutating samples, just samples new data all the time.
	 * In order to merge p and q, simply randomly select p and q for each
	 * feature object inernally */
	CStreamingFeatures* p=m_streaming_p;
	CStreamingFeatures* q=m_streaming_q;
	SG_REF(p);
	SG_REF(q);
	for (index_t i=0; i<m_bootstrap_iterations; ++i)
	{
		/* merge samples by randomly shuffling p and q */
		if (CMath::random(0,1))
			m_streaming_p=p;
		else
			m_streaming_p=q;

		if (CMath::random(0,1))
			m_streaming_q=p;
		else
			m_streaming_q=q;

		/* compute statistic for this permutation of mixed samples */
		samples[i]=compute_statistic();
	}
	m_streaming_p=p;
	m_streaming_q=q;
	SG_UNREF(p);
	SG_UNREF(q);

	return samples;
}

