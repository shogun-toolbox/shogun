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
	m_streaming_p=NULL;
	m_streaming_q=NULL;
	m_blocksize=10000;

	SG_WARNING("%s::init(): register params!\n", get_name());
}

void CLinearTimeMMD::compute_statistic_and_variance(
		SGVector<float64_t>& statistic, SGVector<float64_t>& variance,
		bool multiple_kernels)
{
	SG_DEBUG("entering %s::compute_statistic_and_variance()\n", get_name());

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

	SG_DEBUG("m_m=%d\n", m_m);

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
			"statistic vector size does not match number of kernels\n",
			get_name());

	REQUIRE(variance.vlen==num_kernels, "%s::compute_statistic_and_variance: "
			"variance vector size does not match number of kernels\n",
			get_name());

	/* temp variable in the algorithm */
	float64_t current;
	float64_t delta;

	/* initialise statistic and variance since they are cumulative */
	statistic.zero();
	variance.zero();

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

		/* if multiple kernels are used, compute all of them on streamed data,
		 * if multiple kernels flag is false, the above loop will be executed
		 * only once */
		CKernel* kernel=m_kernel;
		if (multiple_kernels)
			kernel=((CCombinedKernel*)m_kernel)->get_first_kernel();

		for (index_t i=0; i<num_kernels; ++i)
		{
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
				/* compute sum of current h terms for current kernel*/
				current=pp[j]+qq[j]-pq[j]-qp[j];

				/* D. Knuth's online variance algorithm for current kernel */
				delta=current-statistic[i];
				statistic[i]=statistic[i]+delta/term_counter++;
				variance[i]=variance[i]+delta*(current-statistic[i]);

				SG_DEBUG("burst: current=%f, delta=%f, statistic=%f, M2=%f, "
						"kernel_idx=%d\n", current, delta, statistic[i],
						variance[i], i);
			}

			/* if multiple kernels should be computed, set next kernel */
			if (multiple_kernels)
			{
				SG_UNREF(kernel);

				/* safe since the number of iterations are from combined kernel */
				kernel=((CCombinedKernel*)m_kernel)->get_next_kernel();
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

	SG_DEBUG("leaving %s::compute_statistic_and_variance()\n", get_name());
}

void CLinearTimeMMD::compute_statistic_and_Q(
		SGVector<float64_t>& statistic, SGMatrix<float64_t>& Q)
{
	SG_DEBUG("entering %s::compute_statistic_and_Q()\n", get_name());

	REQUIRE(m_streaming_p, "%s::compute_statistic_and_Q: streaming "
			"features p required!\n", get_name());
	REQUIRE(m_streaming_q, "%s::compute_statistic_and_Q: streaming "
			"features q required!\n", get_name());

	REQUIRE(m_kernel, "%s::compute_statistic_and_Q: kernel needed!\n",
			get_name());

	/* make sure multiple_kernels flag is used only with a combined kernel */
	REQUIRE(!m_kernel->get_kernel_type()==K_COMBINED,
			"%s::compute_statistic_and_Q: underlying kernel is not of "
			"type K_COMBINED\n", get_name());

	/* cast combined kernel */
	CCombinedKernel* combined=(CCombinedKernel*)m_kernel;

	/* m is number of samples from each distribution, m_4 is quarter of it */
	REQUIRE(m_m>=4, "%s::compute_statistic_and_Q: Need at least m>=4\n",
			get_name());
	index_t m_4=m_m/4;

	SG_DEBUG("m_m=%d\n", m_m);

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
			"statistic vector size does not match number of kernels\n",
			get_name());

	REQUIRE(Q.num_rows==num_kernels, "%s::compute_statistic_and_variance: "
			"Q number of rows does not match number of kernels\n",
			get_name());

	REQUIRE(Q.num_cols==num_kernels, "%s::compute_statistic_and_variance: "
			"Q number of columns does not match number of kernels\n",
			get_name());

	/* initialise statistic and variance since they are cumulative */
	statistic.zero();
	Q.zero();

	index_t num_examples_processed=0;
	index_t term_counter=1;
	while (num_examples_processed<m_4)
	{
		/* number of example to look at in this iteration */
		index_t num_this_run=CMath::min(m_blocksize, m_4-num_examples_processed);
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
		SG_REF(p1a);
		SG_REF(p1b);
		SG_REF(p2a);
		SG_REF(p2b);
		SG_REF(q1a);
		SG_REF(q1b);
		SG_REF(q2a);
		SG_REF(q2b);

		/* now for each of these streamed data instances, iterate through all
		 * kernels and update Q matrix while also computing MMD statistic */

		/* produce two kernel lists to iterate doubly nested */
		CList* list_i=new CList();
		CList* list_j=new CList();
		CKernel* kernel=combined->get_first_kernel();
		for (index_t i=0; i<num_kernels; ++i)
		{
			list_i->append_element(kernel);
			list_j->append_element(kernel);

			SG_UNREF(kernel);
			kernel=((CCombinedKernel*)m_kernel)->get_next_kernel();
		}

		/* iterate through all kernel pairs for current data */
		CKernel* kernel_i=(CKernel*)list_i->get_first_element();
		CKernel* kernel_j=(CKernel*)list_j->get_first_element();

		/* preallocate some memory for faster processing */
		SGVector<float64_t> pp(num_this_run);
		SGVector<float64_t> qq(num_this_run);
		SGVector<float64_t> pq(num_this_run);
		SGVector<float64_t> qp(num_this_run);
		SGVector<float64_t> h_i_a;
		SGVector<float64_t> h_i_b;
		SGVector<float64_t> h_j_a;
		SGVector<float64_t> h_j_b;

		/* iterate through Q matrix and update values, compute mmd */
		for (index_t i=0; i<num_kernels; ++i)
		{
			for (index_t j=0; j<num_kernels; ++j)
			{
				/* compute all necessary 8 h-vectors for this burst.
				 * h_delta-terms for each kernel, expression 7 of NIPS paper */

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

				/* current term, expression 7 of NIPS paper */
				SGVector<float64_t> term(num_this_run);
				for (index_t it=0; it<num_this_run; ++it)
					term[it]=(h_i_a[it]-h_i_b[it])*(h_j_a[it]-h_j_b[it]);

				/* update covariance element for the current burst. This is a
				 * runnung average of the product of the h_delta terms of each
				 * kernel */
				for (index_t it=0; it<num_this_run; ++i)
					Q(i,j)=Q(i,j)+(term[it]-Q(i,j)/term_counter);

				/* next kernel j */
				SG_UNREF(kernel_j);
				kernel_j=(CKernel*)list_j->get_next_element();
			}

			/* online update of mmd statistic */
//			statistic[i]=statistic[i]+(h-statistic[i])/(i+1);

			/* next kernel i */
			SG_UNREF(kernel_i);
			kernel_i=(CKernel*)list_i->get_next_element();
		}

		/* clean up */
		SG_UNREF(list_i);
		SG_UNREF(list_j);

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
		term_counter++;
	}
	SG_DEBUG("Done compouting statistic, processed 4*%d examples.\n",
			num_examples_processed);

	SG_DEBUG("leaving %s::compute_statistic_and_Q()\n", get_name());
}

//SGVector<float64_t> CLinearTimeMMD::compute_h_terms(bool multiple_kernels)
//{
//	SG_DEBUG("entering %s::compute_h_terms()\n", get_name());
//
//	REQUIRE(m_streaming_p, "%s::compute_h_terms: streaming "
//			"features p required!\n", get_name());
//	REQUIRE(m_streaming_q, "%s::compute_h_terms: streaming "
//			"features q required!\n", get_name());
//
//	REQUIRE(m_kernel, "%s::compute_h_terms: kernel needed!\n",
//			get_name());
//
//	/* the method is basically the same as compute_variance_and_statistic(),
//	 * however, it does not sum up but rather store all terms for the MMD */
//
//	/* m is number of samples from each distribution, m_2 is half of it
//	 * using names from JLMR paper (see class documentation) */
//	index_t m_2=m_m/2;
//	SG_DEBUG("m_m=%d\n", m_m);
//
//	/* allocate space for result */
//	SGVector<float64_t> h(m_2);
//
//	/* these sums are needed to compute online statistic/variance */
//	index_t num_examples_processed=0;
//	while (num_examples_processed<m_2)
//	{
//		/* number of example to look at in this iteration */
//		index_t num_this_run=CMath::min(m_blocksize, m_2-num_examples_processed);
//		SG_DEBUG("processing %d more examples. %d so far processed. Blocksize "
//				"is %d\n", num_this_run, num_examples_processed, m_blocksize);
//
//		/* stream data from both distributions */
//		CFeatures* p1=m_streaming_p->get_streamed_features(num_this_run);
//		CFeatures* p2=m_streaming_p->get_streamed_features(num_this_run);
//		CFeatures* q1=m_streaming_q->get_streamed_features(num_this_run);
//		CFeatures* q2=m_streaming_q->get_streamed_features(num_this_run);
//		SG_REF(p1);
//		SG_REF(p2);
//		SG_REF(q1);
//		SG_REF(q2);
//
//		/* compute kernel matrix diagonals */
//		SG_DEBUG("processing kernel diagonal pp\n");
//		m_kernel->init(p1, p2);
//		SGVector<float64_t> pp=m_kernel->get_kernel_diagonal();
//
//		SG_DEBUG("processing kernel diagonal qq\n");
//		m_kernel->init(q1, q2);
//		SGVector<float64_t> qq=m_kernel->get_kernel_diagonal();
//
//		SG_DEBUG("processing kernel diagonal pq\n");
//		m_kernel->init(p1, q2);
//		SGVector<float64_t> pq=m_kernel->get_kernel_diagonal();
//
//		SG_DEBUG("processing kernel diagonal qp\n");
//		m_kernel->init(q1, p2);
//		SGVector<float64_t> qp=m_kernel->get_kernel_diagonal();
//
//		/* fill in processed part of h-term for current kernel */
//		for (index_t j=0; j<pp.vlen; ++j)
//			h[num_examples_processed+j]=pp[j]+qq[j]-pq[j]-qp[j];
//
//		/* clean up */
//		SG_UNREF(p1);
//		SG_UNREF(p2);
//		SG_UNREF(q1);
//		SG_UNREF(q2);
//
//		/* add number of processed examples for this run */
//		num_examples_processed+=num_this_run;
//	}
//	SG_DEBUG("Done compouting h-terms, processed 2*%d examples.\n",
//			num_examples_processed);
//
//	SG_WARNING("%s::compute_h_terms(): Not yet tested!\n");
//
//	SG_DEBUG("leaving %s::compute_h_terms()\n", get_name());
//	return h;
//}

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

