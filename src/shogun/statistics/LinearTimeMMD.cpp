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
	SG_ADD(&m_opt_max_iterations, "opt_max_iterations", "Maximum number of "
			"iterations for qp solver", MS_NOT_AVAILABLE);
	SG_ADD(&m_opt_epsilon, "opt_epsilon", "Stopping criterion for qp solver",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_opt_low_cut, "opt_low_cut", "Low cut value for optimization "
			"kernel weights", MS_NOT_AVAILABLE);
	SG_ADD(&m_opt_regularization_eps, "opt_regularization_eps", "Regularization"
			" value that is added to diagonal of Q matrix", MS_NOT_AVAILABLE);

	m_opt_max_iterations=10000;
	m_opt_epsilon=10E-15;
	m_opt_low_cut=10E-7;
	m_opt_regularization_eps=0;
	m_streaming_p=NULL;
	m_streaming_q=NULL;
	m_blocksize=10000;

	SG_WARNING("%s::init(): register params!\n", get_name());
}

void CLinearTimeMMD::compute_statistic_and_variance(
		float64_t& statistic, float64_t& variance)
{
	SG_DEBUG("entering CLinearTimeMMD::compute_statistic_and_variance()\n");

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

	SG_DEBUG("leaving CLinearTimeMMD::compute_statistic_and_variance()\n");
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

#ifdef HAVE_LAPACK
void CLinearTimeMMD::optimize_kernel_weights()
{
	/** TODO check whether other types of combined kernels/features might be
	 * allowed */
	if (m_kernel->get_kernel_type()!=K_COMBINED)
	{
		SG_ERROR("CLinearTimeMMD::optimize_kernel_weights(): Only possible "
				"with a combined kernel!\n");
	}

	if (m_p_and_q->get_feature_class()!=C_COMBINED)
	{
		SG_ERROR("CLinearTimeMMD::optimize_kernel_weights(): Only possible "
				"with combined features!\n");
	}

	/* think about casting and types of different kernels/features here */
	CCombinedFeatures* combined_p_and_q=
			dynamic_cast<CCombinedFeatures*>(m_p_and_q);
	CCombinedKernel* combined_kernel=dynamic_cast<CCombinedKernel*>(m_kernel);
	ASSERT(combined_p_and_q);
	ASSERT(combined_kernel);

	if (combined_kernel->get_num_subkernels()!=
			combined_p_and_q->get_num_feature_obj())
	{
		SG_ERROR("CLinearTimeMMD::optimize_kernel_weights(): Only possible "
				"when number of sub-kernels (%d) equal number of sub-features "
				"(%d)\n", combined_kernel->get_num_subkernels(),
				combined_p_and_q->get_num_feature_obj());
	}

	/* init kernel with features */
	m_kernel->init(m_p_and_q, m_p_and_q);

	/* number of kernels and data */
	index_t num_kernels=combined_kernel->get_num_subkernels();
	index_t m2=m_m/2;

	/* matrix with all h entries for all kernels and data */
	SGMatrix<float64_t> hs(m2, num_kernels);

	/* mmds are needed and are means of columns of hs */
	SGVector<float64_t> mmds(num_kernels);

	float64_t pp;
	float64_t qq;
	float64_t pq;
	float64_t qp;
	/* compute all h entries */
	for (index_t i=0; i<num_kernels; ++i)
	{
		CKernel* current=combined_kernel->get_kernel(i);
		mmds[i]=0;
		for (index_t j=0; j<m2; ++j)
		{
			pp=current->kernel(j, m2+j);
			qq=current->kernel(m_m+j, m_m+m2+j);
			pq=current->kernel(j, m_m+m2+j);
			qp=current->kernel(m2+j, m_m+j);
			hs(j, i)=pp+qq-pq-qp;
			mmds[i]+=hs(j, i);
		}

		/* mmd is simply mean. This is the unbiased linear time estimate */
		mmds[i]/=m2;

		SG_UNREF(current);
	}

	/* compute covariance matrix of h vector, in place is safe now since h
	 * is not needed anymore */
	m_Q=CStatistics::covariance_matrix(hs, true);

	/* evtl regularize to avoid numerical problems (ratio of MMD and std-dev
	 * blows up when variance is small */
	if (m_opt_regularization_eps)
	{
		SG_DEBUG("regularizing matrix Q by adding %f to diagonal\n",
				m_opt_regularization_eps);
		for (index_t i=0; i<num_kernels; ++i)
			m_Q(i,i)+=m_opt_regularization_eps;
	}

	if (sg_io->get_loglevel()==MSG_DEBUG)
	{
		m_Q.display_matrix("(evtl. regularized) Q");
		mmds.display_vector("mmds");
	}

	/* compute sum of mmds to generate feasible point for convex program */
	float64_t sum_mmds=0;
	for (index_t i=0; i<mmds.vlen; ++i)
		sum_mmds+=mmds[i];

	/* QP: 0.5*x'*Q*x + f'*x
	 * subject to
	 * mmds'*x = b
	 * LB[i] <= x[i] <= UB[i]   for all i=1..n */
	SGVector<float64_t> Q_diag(num_kernels);
	SGVector<float64_t> f(num_kernels);
	SGVector<float64_t> lb(num_kernels);
	SGVector<float64_t> ub(num_kernels);
	SGVector<float64_t> x(num_kernels);

	/* init everything, there are two cases possible: i) at least one mmd is
	 * is positive, ii) all mmds are negative */
	bool one_pos;
	for (index_t i=0; i<mmds.vlen; ++i)
	{
		if (mmds[i]>0)
		{
			SG_DEBUG("found at least one positive MMD\n");
			one_pos=true;
			break;
		}
		one_pos=false;
	}

	if (!one_pos)
	{
		SG_WARNING("All mmd estimates are negative. This is techical possible,"
				" although extremely rare. Current problem might bad\n");

		/* if no element is positive, Q has to be replaced by -Q */
		for (index_t i=0; i<num_kernels*num_kernels; ++i)
			m_Q.matrix[i]=-m_Q.matrix[i];
	}

	/* init vectors */
	for (index_t i=0; i<num_kernels; ++i)
	{
		Q_diag[i]=m_Q(i,i);
		f[i]=0;
		lb[i]=0;
		ub[i]=CMath::INFTY;

		/* initial point has to be feasible, i.e. mmds'*x = b */
		x[i]=1.0/sum_mmds;
	}

	/* start libqp solver with desired parameters */
	SG_DEBUG("starting libqp optimization\n");
	libqp_state_T qp_exitflag=libqp_gsmo_solver(&get_Q_col, Q_diag.vector,
			f.vector, mmds.vector,
			one_pos ? 1 : -1,
			lb.vector, ub.vector,
			x.vector, num_kernels, m_opt_max_iterations,
			m_opt_regularization_eps, &print_state);

	SG_DEBUG("libqp returns: nIts=%d, exit_flag: %d\n", qp_exitflag.nIter,
			qp_exitflag.exitflag);

	/* set really small entries to zero and sum up for normalization */
	float64_t sum_weights=0;
	for (index_t i=0; i<x.vlen; ++i)
	{
		if (x[i]<m_opt_low_cut)
		{
			SG_DEBUG("lowcut: weight[%i]=%f<%f; setting to zero\n", i, x[i],
					m_opt_low_cut);
			x[i]=0;
		}

		sum_weights+=x[i];
	}

	/* normalize (allowed since problem is scale invariant) */
	for (index_t i=0; i<x.vlen; ++i)
		x[i]/=sum_weights;

	/* set weights to kernel */
	m_kernel->set_subkernel_weights(x);
}

SGMatrix<float64_t> CLinearTimeMMD::m_Q=SGMatrix<float64_t>();

const float64_t* CLinearTimeMMD::get_Q_col(uint32_t i)
{
	return &m_Q[m_Q.num_rows*i];
}

void CLinearTimeMMD::print_state(libqp_state_T state)
{
	SG_SDEBUG("libqp state: primal=%f\n", state.QP);
}

#endif //HAVE_LAPACK

