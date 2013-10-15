/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/features/Features.h>
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/lib/List.h>

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
	SG_ADD((CSGObject**)&m_streaming_p, "streaming_p", "Streaming features p",
				MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_streaming_q, "streaming_q", "Streaming features p",
				MS_NOT_AVAILABLE);
	SG_ADD(&m_blocksize, "blocksize", "Number of elements processed at once",
				MS_NOT_AVAILABLE);
	SG_ADD(&m_simulate_h0, "simulate_h0", "Whether p and q are mixed",
				MS_NOT_AVAILABLE);

	m_streaming_p=NULL;
	m_streaming_q=NULL;
	m_blocksize=10000;
	m_simulate_h0=false;
}

void CLinearTimeMMD::compute_statistic_and_variance(
		SGVector<float64_t>& statistic, SGVector<float64_t>& variance,
		bool multiple_kernels)
{
	SG_DEBUG("entering %s::compute_statistic_and_variance()\n", get_name())

	REQUIRE(m_streaming_p, "%s::compute_statistic_and_variance: streaming "
			"features p required!\n", get_name());
	REQUIRE(m_streaming_q, "%s::compute_statistic_and_variance: streaming "
			"features q required!\n", get_name());

	REQUIRE(m_kernel, "%s::compute_statistic_and_variance: kernel needed!\n",
			get_name());

	/* make sure multiple_kernels flag is used only with a combined kernel */
	REQUIRE(!multiple_kernels || m_kernel->get_kernel_type()==K_COMBINED,
			"%s::compute_statistic_and_variance: multiple kernels specified,"
			"but underlying kernel is not of type K_COMBINED\n", get_name());

	/* m is number of samples from each distribution, m_2 is half of it
	 * using names from JLMR paper (see class documentation) */
	index_t m_2=m_m/2;

	SG_DEBUG("m_m=%d\n", m_m)

	/* find out whether single or multiple kernels (cast is safe, check above) */
	index_t num_kernels=1;
	if (multiple_kernels)
	{
		num_kernels=((CCombinedKernel*)m_kernel)->get_num_subkernels();
		SG_DEBUG("computing MMD and variance for %d sub-kernels\n",
				num_kernels);
	}

	/* allocate memory for results if vectors are empty */
	if (!statistic.vector)
		statistic=SGVector<float64_t>(num_kernels);

	if (!variance.vector)
		variance=SGVector<float64_t>(num_kernels);

	/* ensure right dimensions */
	REQUIRE(statistic.vlen==num_kernels, "%s::compute_statistic_and_variance: "
			"statistic vector size (%d) does not match number of kernels (%d)\n",
			 get_name(), statistic.vlen, num_kernels);

	REQUIRE(variance.vlen==num_kernels, "%s::compute_statistic_and_variance: "
			"variance vector size (%d) does not match number of kernels (%d)\n",
			 get_name(), variance.vlen, num_kernels);

	/* temp variable in the algorithm */
	float64_t current;
	float64_t delta;

	/* initialise statistic and variance since they are cumulative */
	statistic.zero();
	variance.zero();

	/* needed for online mean and variance */
	SGVector<index_t> term_counters(num_kernels);
	term_counters.set_const(1);

	/* term counter to compute online mean and variance */
	index_t num_examples_processed=0;
	while (num_examples_processed<m_2)
	{
		/* number of example to look at in this iteration */
		index_t num_this_run=CMath::min(m_blocksize,
				CMath::max(0, m_2-num_examples_processed));
		SG_DEBUG("processing %d more examples. %d so far processed. Blocksize "
				"is %d\n", num_this_run, num_examples_processed, m_blocksize);

		/* stream data from both distributions */
		CFeatures* p1=m_streaming_p->get_streamed_features(num_this_run);
		CFeatures* p2=m_streaming_p->get_streamed_features(num_this_run);
		CFeatures* q1=m_streaming_q->get_streamed_features(num_this_run);
		CFeatures* q2=m_streaming_q->get_streamed_features(num_this_run);

		/* check whether h0 should be simulated and permute if so */
		if (m_simulate_h0)
		{
			/* create merged copy of all feature instances to permute */
			CList* list=new CList();
			list->append_element(p2);
			list->append_element(q1);
			list->append_element(q2);
			CFeatures* merged=p1->create_merged_copy(list);
			SG_UNREF(list);

			/* permute */
			SGVector<index_t> inds(merged->get_num_vectors());
			inds.range_fill();
			inds.permute();
			merged->add_subset(inds);

			/* copy back, replacing old features */
			SG_UNREF(p1);
			SG_UNREF(p2);
			SG_UNREF(q1);
			SG_UNREF(q2);

			SGVector<index_t> copy(num_this_run);
			copy.range_fill();
			p1=merged->copy_subset(copy);
			copy.add(num_this_run);
			p2=merged->copy_subset(copy);
			copy.add(num_this_run);
			q1=merged->copy_subset(copy);
			copy.add(num_this_run);
			q2=merged->copy_subset(copy);

			/* clean up and note that copy_subset does a SG_REF */
			SG_UNREF(merged);
		}
		else
		{
			/* reference produced features (only if copy_subset was not used) */
			SG_REF(p1);
			SG_REF(p2);
			SG_REF(q1);
			SG_REF(q2);
		}

		/* if multiple kernels are used, compute all of them on streamed data,
		 * if multiple kernels flag is false, the above loop will be executed
		 * only once */
		CKernel* kernel=m_kernel;
		if (multiple_kernels)
		{
			SG_DEBUG("using multiple kernels\n");
		}

		/* iterate through all kernels for this data */
		for (index_t i=0; i<num_kernels; ++i)
		{
			/* if multiple kernels should be computed, set next kernel */
			if (multiple_kernels)
			{
				kernel=((CCombinedKernel*)m_kernel)->get_kernel(i);
			}

			/* compute kernel matrix diagonals */
			kernel->init(p1, p2);
			SGVector<float64_t> pp=kernel->get_kernel_diagonal();

			kernel->init(q1, q2);
			SGVector<float64_t> qq=kernel->get_kernel_diagonal();

			kernel->init(p1, q2);
			SGVector<float64_t> pq=kernel->get_kernel_diagonal();

			kernel->init(q1, p2);
			SGVector<float64_t> qp=kernel->get_kernel_diagonal();

			/* single variances for all kernels. Update mean and variance
			 * using Knuth's online variance algorithm.
			 * C.f. for example Wikipedia */
			for (index_t j=0; j<num_this_run; ++j)
			{
				/* compute sum of current h terms for current kernel */
				current=pp[j]+qq[j]-pq[j]-qp[j];

				/* D. Knuth's online variance algorithm for current kernel */
				delta=current-statistic[i];
				statistic[i]+=delta/term_counters[i]++;
				variance[i]+=delta*(current-statistic[i]);

				SG_DEBUG("burst: current=%f, delta=%f, statistic=%f, "
						"variance=%f, kernel_idx=%d\n", current, delta,
						statistic[i], variance[i], i);
			}

			if (multiple_kernels)
			{
				SG_UNREF(kernel);
			}
		}

		/* clean up streamed data */
		SG_UNREF(p1);
		SG_UNREF(p2);
		SG_UNREF(q1);
		SG_UNREF(q2);

		/* add number of processed examples for this run */
		num_examples_processed+=num_this_run;
	}
	SG_DEBUG("Done compouting statistic, processed 2*%d examples.\n",
			num_examples_processed);

	/* mean of sum all traces is linear time mmd, copy entries for all kernels */
	if (io->get_loglevel()==MSG_DEBUG || io->get_loglevel()==MSG_GCDEBUG)
		statistic.display_vector("statistics");

	/* variance of terms can be computed using mean (statistic).
	 * Note that the variance needs to be divided by m_2 in order to get
	 * variance of null-distribution */
	for (index_t i=0; i<num_kernels; ++i)
		variance[i]=variance[i]/(m_2-1)/m_2;

	if (io->get_loglevel()==MSG_DEBUG || io->get_loglevel()==MSG_GCDEBUG)
		variance.display_vector("variances");

	SG_DEBUG("leaving %s::compute_statistic_and_variance()\n", get_name())
}

void CLinearTimeMMD::compute_statistic_and_Q(
		SGVector<float64_t>& statistic, SGMatrix<float64_t>& Q)
{
	SG_DEBUG("entering %s::compute_statistic_and_Q()\n", get_name())

	REQUIRE(m_streaming_p, "%s::compute_statistic_and_Q: streaming "
			"features p required!\n", get_name());
	REQUIRE(m_streaming_q, "%s::compute_statistic_and_Q: streaming "
			"features q required!\n", get_name());

	REQUIRE(m_kernel, "%s::compute_statistic_and_Q: kernel needed!\n",
			get_name());

	/* make sure multiple_kernels flag is used only with a combined kernel */
	REQUIRE(m_kernel->get_kernel_type()==K_COMBINED,
			"%s::compute_statistic_and_Q: underlying kernel is not of "
			"type K_COMBINED\n", get_name());

	/* cast combined kernel */
	CCombinedKernel* combined=(CCombinedKernel*)m_kernel;

	/* m is number of samples from each distribution, m_4 is quarter of it */
	REQUIRE(m_m>=4, "%s::compute_statistic_and_Q: Need at least m>=4\n",
			get_name());
	index_t m_4=m_m/4;

	SG_DEBUG("m_m=%d\n", m_m)

	/* find out whether single or multiple kernels (cast is safe, check above) */
	index_t num_kernels=combined->get_num_subkernels();
	REQUIRE(num_kernels>0, "%s::compute_statistic_and_Q: At least one kernel "
			"is needed\n", get_name());

	/* allocate memory for results if vectors are empty */
	if (!statistic.vector)
		statistic=SGVector<float64_t>(num_kernels);

	if (!Q.matrix)
		Q=SGMatrix<float64_t>(num_kernels, num_kernels);

	/* ensure right dimensions */
	REQUIRE(statistic.vlen==num_kernels, "%s::compute_statistic_and_variance: "
			"statistic vector size (%d) does not match number of kernels (%d)\n",
			 get_name(), statistic.vlen, num_kernels);

	REQUIRE(Q.num_rows==num_kernels, "%s::compute_statistic_and_variance: "
			"Q number of rows does (%d) not match number of kernels (%d)\n",
			 get_name(), Q.num_rows, num_kernels);

	REQUIRE(Q.num_cols==num_kernels, "%s::compute_statistic_and_variance: "
			"Q number of columns (%d) does not match number of kernels (%d)\n",
			 get_name(), Q.num_cols, num_kernels);

	/* initialise statistic and variance since they are cumulative */
	statistic.zero();
	Q.zero();

	/* produce two kernel lists to iterate doubly nested */
	CList* list_i=new CList();
	CList* list_j=new CList();

	for (index_t k_idx=0; k_idx<combined->get_num_kernels(); k_idx++)
	{
		CKernel* kernel = combined->get_kernel(k_idx);
		list_i->append_element(kernel);
		list_j->append_element(kernel);
		SG_UNREF(kernel);
	}

	/* needed for online mean and variance */
	SGVector<index_t> term_counters_statistic(num_kernels);
	SGMatrix<index_t> term_counters_Q(num_kernels, num_kernels);
	term_counters_statistic.set_const(1);
	term_counters_Q.set_const(1);

	index_t num_examples_processed=0;
	while (num_examples_processed<m_4)
	{
		/* number of example to look at in this iteration */
		index_t num_this_run=CMath::min(m_blocksize,
				CMath::max(0, m_4-num_examples_processed));
		SG_DEBUG("processing %d more examples. %d so far processed. Blocksize "
				"is %d\n", num_this_run, num_examples_processed, m_blocksize);

		/* stream data from both distributions */
		CFeatures* p1a=m_streaming_p->get_streamed_features(num_this_run);
		CFeatures* p1b=m_streaming_p->get_streamed_features(num_this_run);
		CFeatures* p2a=m_streaming_p->get_streamed_features(num_this_run);
		CFeatures* p2b=m_streaming_p->get_streamed_features(num_this_run);
		CFeatures* q1a=m_streaming_q->get_streamed_features(num_this_run);
		CFeatures* q1b=m_streaming_q->get_streamed_features(num_this_run);
		CFeatures* q2a=m_streaming_q->get_streamed_features(num_this_run);
		CFeatures* q2b=m_streaming_q->get_streamed_features(num_this_run);

		/* check whether h0 should be simulated and permute if so */
		if (m_simulate_h0)
		{
			/* create merged copy of all feature instances to permute */
			CList* list=new CList();
			list->append_element(p1b);
			list->append_element(p2a);
			list->append_element(p2b);
			list->append_element(q1a);
			list->append_element(q1b);
			list->append_element(q2a);
			list->append_element(q2b);
			CFeatures* merged=p1a->create_merged_copy(list);
			SG_UNREF(list);

			/* permute */
			SGVector<index_t> inds(merged->get_num_vectors());
			inds.range_fill();
			inds.permute();
			merged->add_subset(inds);

			/* copy back, replacing old features */
			SG_UNREF(p1a);
			SG_UNREF(p1b);
			SG_UNREF(p2a);
			SG_UNREF(p2b);
			SG_UNREF(q1a);
			SG_UNREF(q1b);
			SG_UNREF(q2a);
			SG_UNREF(q2b);

			SGVector<index_t> copy(num_this_run);
			copy.range_fill();
			p1a=merged->copy_subset(copy);
			copy.add(num_this_run);
			p1b=merged->copy_subset(copy);
			copy.add(num_this_run);
			p2a=merged->copy_subset(copy);
			copy.add(num_this_run);
			p2b=merged->copy_subset(copy);
			copy.add(num_this_run);
			q1a=merged->copy_subset(copy);
			copy.add(num_this_run);
			q1b=merged->copy_subset(copy);
			copy.add(num_this_run);
			q2a=merged->copy_subset(copy);
			copy.add(num_this_run);
			q2b=merged->copy_subset(copy);

			/* clean up and note that copy_subset does a SG_REF */
			SG_UNREF(merged);
		}
		else
		{
			/* reference the produced features (only if copy subset was not used) */
			SG_REF(p1a);
			SG_REF(p1b);
			SG_REF(p2a);
			SG_REF(p2b);
			SG_REF(q1a);
			SG_REF(q1b);
			SG_REF(q2a);
			SG_REF(q2b);
		}

		/* now for each of these streamed data instances, iterate through all
		 * kernels and update Q matrix while also computing MMD statistic */

		/* preallocate some memory for faster processing */
		SGVector<float64_t> pp(num_this_run);
		SGVector<float64_t> qq(num_this_run);
		SGVector<float64_t> pq(num_this_run);
		SGVector<float64_t> qp(num_this_run);
		SGVector<float64_t> h_i_a(num_this_run);
		SGVector<float64_t> h_i_b(num_this_run);
		SGVector<float64_t> h_j_a(num_this_run);
		SGVector<float64_t> h_j_b(num_this_run);

		/* iterate through Q matrix and update values, compute mmd */
		CKernel* kernel_i=(CKernel*)list_i->get_first_element();
		for (index_t i=0; i<num_kernels; ++i)
		{
			/* compute all necessary 8 h-vectors for this burst.
			 * h_delta-terms for each kernel, expression 7 of NIPS paper
			 * first kernel */

			/* first kernel, a-part */
			kernel_i->init(p1a, p2a);
			pp=kernel_i->get_kernel_diagonal(pp);
			kernel_i->init(q1a, q2a);
			qq=kernel_i->get_kernel_diagonal(qq);
			kernel_i->init(p1a, q2a);
			pq=kernel_i->get_kernel_diagonal(pq);
			kernel_i->init(q1a, p2a);
			qp=kernel_i->get_kernel_diagonal(qp);
			for (index_t it=0; it<num_this_run; ++it)
				h_i_a[it]=pp[it]+qq[it]-pq[it]-qp[it];

			/* first kernel, b-part */
			kernel_i->init(p1b, p2b);
			pp=kernel_i->get_kernel_diagonal(pp);
			kernel_i->init(q1b, q2b);
			qq=kernel_i->get_kernel_diagonal(qq);
			kernel_i->init(p1b, q2b);
			pq=kernel_i->get_kernel_diagonal(pq);
			kernel_i->init(q1b, p2b);
			qp=kernel_i->get_kernel_diagonal(qp);
			for (index_t it=0; it<num_this_run; ++it)
				h_i_b[it]=pp[it]+qq[it]-pq[it]-qp[it];

			/* iterate through j, but use symmetry in order to save half of the
			 * computations */
			CKernel* kernel_j=(CKernel*)list_j->get_first_element();
			for (index_t j=0; j<=i; ++j)
			{
				/* compute all necessary 8 h-vectors for this burst.
				 * h_delta-terms for each kernel, expression 7 of NIPS paper
				 * second kernel */

				/* second kernel, a-part */
				kernel_j->init(p1a, p2a);
				pp=kernel_j->get_kernel_diagonal(pp);
				kernel_j->init(q1a, q2a);
				qq=kernel_j->get_kernel_diagonal(qq);
				kernel_j->init(p1a, q2a);
				pq=kernel_j->get_kernel_diagonal(pq);
				kernel_j->init(q1a, p2a);
				qp=kernel_j->get_kernel_diagonal(qp);
				for (index_t it=0; it<num_this_run; ++it)
					h_j_a[it]=pp[it]+qq[it]-pq[it]-qp[it];

				/* second kernel, b-part */
				kernel_j->init(p1b, p2b);
				pp=kernel_j->get_kernel_diagonal(pp);
				kernel_j->init(q1b, q2b);
				qq=kernel_j->get_kernel_diagonal(qq);
				kernel_j->init(p1b, q2b);
				pq=kernel_j->get_kernel_diagonal(pq);
				kernel_j->init(q1b, p2b);
				qp=kernel_j->get_kernel_diagonal(qp);
				for (index_t it=0; it<num_this_run; ++it)
					h_j_b[it]=pp[it]+qq[it]-pq[it]-qp[it];

				float64_t term;
				for (index_t it=0; it<num_this_run; ++it)
				{
					/* current term of expression 7 of NIPS paper */
					term=(h_i_a[it]-h_i_b[it])*(h_j_a[it]-h_j_b[it]);

					/* update covariance element for the current burst. This is a
					 * running average of the product of the h_delta terms of each
					 * kernel */
					Q(i, j)+=(term-Q(i, j))/term_counters_Q(i, j)++;
				}

				/* use symmetry */
				Q(j, i)=Q(i, j);

				/* next kernel j */
				kernel_j=(CKernel*)list_j->get_next_element();
			}

			/* update MMD statistic online computation for kernel i, using
			 * vectors that were computed above */
			SGVector<float64_t> h(num_this_run*2);
			for (index_t it=0; it<num_this_run; ++it)
			{
				/* update statistic for kernel i (outer loop) and update using
				 * all elements of the h_i_a, h_i_b vectors (iterate over it) */
				statistic[i]=statistic[i]+
						(h_i_a[it]-statistic[i])/term_counters_statistic[i]++;

				/* Make sure to use all data, i.e. part a and b */
				statistic[i]=statistic[i]+
						(h_i_b[it]-statistic[i])/(term_counters_statistic[i]++);
			}

			/* next kernel i */
			kernel_i=(CKernel*)list_i->get_next_element();
		}

		/* clean up streamed data */
		SG_UNREF(p1a);
		SG_UNREF(p1b);
		SG_UNREF(p2a);
		SG_UNREF(p2b);
		SG_UNREF(q1a);
		SG_UNREF(q1b);
		SG_UNREF(q2a);
		SG_UNREF(q2b);

		/* add number of processed examples for this run */
		num_examples_processed+=num_this_run;
	}

	/* clean up */
	SG_UNREF(list_i);
	SG_UNREF(list_j);

	SG_DEBUG("Done compouting statistic, processed 4*%d examples.\n",
			num_examples_processed);

	SG_DEBUG("leaving %s::compute_statistic_and_Q()\n", get_name())
}

float64_t CLinearTimeMMD::compute_statistic()
{
	/* use wrapper method and compute for single kernel */
	SGVector<float64_t> statistic;
	SGVector<float64_t> variance;
	compute_statistic_and_variance(statistic, variance, false);

	return statistic[0];
}

SGVector<float64_t> CLinearTimeMMD::compute_statistic(
		bool multiple_kernels)
{
	/* make sure multiple_kernels flag is used only with a combined kernel */
	REQUIRE(!multiple_kernels || m_kernel->get_kernel_type()==K_COMBINED,
			"%s::compute_statistic: multiple kernels specified,"
			"but underlying kernel is not of type K_COMBINED\n", get_name());

	SGVector<float64_t> statistic;
	SGVector<float64_t> variance;
	compute_statistic_and_variance(statistic, variance, multiple_kernels);

	return statistic;
}

float64_t CLinearTimeMMD::compute_variance_estimate()
{
	/* use wrapper method and compute for single kernel */
	SGVector<float64_t> statistic;
	SGVector<float64_t> variance;
	compute_statistic_and_variance(statistic, variance, false);

	return variance[0];
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
			/* compute variance and use to estimate Gaussian distribution, use
			 * wrapper method and compute for single kernel */
			SGVector<float64_t> statistic;
			SGVector<float64_t> variance;
			compute_statistic_and_variance(statistic, variance, false);

			/* estimate Gaussian distribution */
			result=1.0-CStatistics::normal_cdf(statistic[0],
					CMath::sqrt(variance[0]));
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

	/* instead of permutating samples, just samples new data all the time. */
	CStreamingFeatures* p=m_streaming_p;
	CStreamingFeatures* q=m_streaming_q;
	SG_REF(p);
	SG_REF(q);

	bool old=m_simulate_h0;
	set_simulate_h0(true);
	for (index_t i=0; i<m_bootstrap_iterations; ++i)
	{
		/* compute statistic for this permutation of mixed samples */
		samples[i]=compute_statistic();
	}
	set_simulate_h0(old);
	m_streaming_p=p;
	m_streaming_q=q;
	SG_UNREF(p);
	SG_UNREF(q);

	return samples;
}

void CLinearTimeMMD::set_p_and_q(CFeatures* p_and_q)
{
	SG_ERROR("%s::set_p_and_q(): Method not implemented since linear time mmd"
			" is based on streaming features\n", get_name());
}

CFeatures* CLinearTimeMMD::get_p_and_q()
{
	SG_ERROR("%s::get_p_and_q(): Method not implemented since linear time mmd"
			" is based on streaming features\n", get_name());
	return NULL;
}

CStreamingFeatures* CLinearTimeMMD::get_streaming_p()
{
	SG_REF(m_streaming_p);
	return m_streaming_p;
}

CStreamingFeatures* CLinearTimeMMD::get_streaming_q()
{
	SG_REF(m_streaming_q);
	return m_streaming_q;
}

