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
#include <shogun/kernel/StreamingKernel.h>

#include <shogun/lib/external/libqp.h>

using namespace shogun;

CLinearTimeMMD::CLinearTimeMMD() :
		CKernelTwoSampleTestStatistic()
{
	init();
}

CLinearTimeMMD::CLinearTimeMMD(CKernel* kernel, CFeatures* p_and_q,
		index_t m)
{
	SG_ERROR("%s::CLinearTimeMMD(): Constructor with appended features is "
			"not supported. Split features and use other constructor.\n",
			get_name());
}

CLinearTimeMMD::CLinearTimeMMD(CKernel* kernel, CFeatures* p, CFeatures* q,
		index_t m)
{
	/* convienience constructor, construct streaming kernel and features from
	 * the provided objects. Thats why base constructor is NOT called */

	/* prevent calling this with streaming features */
	CStreamingFeatures* temp1=dynamic_cast<CStreamingFeatures*>(p);
	CStreamingFeatures* temp2=dynamic_cast<CStreamingFeatures*>(q);
	CStreamingKernel* temp3=dynamic_cast<CStreamingKernel*>(kernel);
	REQUIRE(!(temp1 || temp2 || temp3), "%s::CLinearTimeMMD(): Please use "
			"streaming features based constructor when passing streaming "
			"features. This constructor can only handle non-streaming features"
			" and kernels!\n", get_name());

	m_m=m;

	/* create streaming features from given features. Same for kernel */
	m_streaming_p=CStreamingFeatures::from_non_streaming(p);
	SG_REF(m_streaming_p);
	m_streaming_p->start_parser();

	m_streaming_q=CStreamingFeatures::from_non_streaming(q);
	SG_REF(m_streaming_q);
	m_streaming_q->start_parser();

	m_kernel=new CStreamingKernel(m_streaming_p, m_streaming_q, kernel);
	SG_REF(m_kernel);
}

CLinearTimeMMD::CLinearTimeMMD(CStreamingKernel* kernel,
		CStreamingFeatures* p, CStreamingFeatures* q, index_t m) :
		CKernelTwoSampleTestStatistic(kernel, NULL, m)
{
	init();

	m_streaming_p=p;
	SG_REF(m_streaming_p);

	m_streaming_q=q;
	SG_REF(m_streaming_q);
}

CLinearTimeMMD::~CLinearTimeMMD()
{
	SG_UNREF(m_streaming_p);
	SG_UNREF(m_streaming_q);
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

	SG_WARNING("register params!\n");
}

float64_t CLinearTimeMMD::compute_statistic()
{
	SG_DEBUG("entering CLinearTimeMMD::compute_statistic()\n");

	REQUIRE(m_streaming_p, "%s::compute_statistic: streaming features p "
			"required!\n", get_name());
	REQUIRE(m_streaming_q, "%s::compute_statistic: streaming features q "
			"required!\n", get_name());

	REQUIRE(m_kernel, "%s::compute_statistic: kernel needed!\n", get_name());

	/* cast kernel as streaming kernel */
	CStreamingKernel* streaming_kernel=
			dynamic_cast<CStreamingKernel*>(m_kernel);
	REQUIRE(streaming_kernel,
			"%s::compute_statistic: streaming kernel required!\n", get_name());

	/* m is number of samples from each distribution, m_2 is half of it
	 * using names from JLMR paper (see class documentation) */
	index_t m_2=m_m/2;

	SG_DEBUG("m_m=%d\n", m_m);

	/* these sums are needed to compute statistic */
	float64_t pp=0;
	float64_t qq=0;
	float64_t pq=0;
	float64_t qp=0;

	/* compute sums */
	for (index_t i=0; i<m_2; ++i)
	{
		/* stream kernel values and add values */
		SG_DEBUG("initialising streaming kernel with pp\n");
		streaming_kernel->init(m_streaming_p, m_streaming_p);
		float64_t temp=streaming_kernel->kernel(0, 0);
		pp+=temp;
		SG_DEBUG("pp+=%f\n", temp);

		SG_DEBUG("initialising streaming kernel with qq\n");
		streaming_kernel->init(m_streaming_q, m_streaming_q);
		temp=streaming_kernel->kernel(0, 0);
		qq+=temp;
		SG_DEBUG("qq+=%f\n", temp);

		SG_DEBUG("initialising streaming kernel with pq\n");
		streaming_kernel->init(m_streaming_p, m_streaming_q);
		temp=streaming_kernel->kernel(0, 0);
		pq+=temp;
		SG_DEBUG("pq+=%f\n", temp);

		temp=streaming_kernel->kernel(0, 0);
		qp+=temp;
		SG_DEBUG("pq+=%f\n", temp);
	}

	SG_DEBUG("returning: 1/%d*(%f+%f-%f-%f)\n", m_2, pp, qq, pq, qp);

	/* mean of sum all traces is linear time mmd */
	SG_DEBUG("leaving CLinearTimeMMD::compute_statistic()\n");
	return 1.0/m_2*(pp+qq-pq-qp);
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

float64_t CLinearTimeMMD::compute_variance_estimate()
{
	REQUIRE(m_p_and_q, "%s::compute_variance_estimate: features needed!\n",
			get_name());

	REQUIRE(m_kernel, "%s::compute_variance_estimate: kernel needed!\n",
			get_name());

	if (m_p_and_q->get_num_vectors()<1000)
	{
		SG_WARNING("%s::compute_variance_estimate: The number of samples"
				" should be very large (at least 1000)  in order to get a"
				" good Gaussian approximation using MMD1_GAUSSIAN.\n",
				get_name());
	}

	/* this corresponds to computing the statistic itself, however, the
	 * difference is that all terms (of the traces) have to be stored */
	index_t m=m_m;
	index_t m_2=m/2;

	m_kernel->init(m_p_and_q, m_p_and_q);

	/* allocate memory for traces */
	SGVector<float64_t> traces(m_2);

	/* sum up diagonals of all kernel matrices */
	for (index_t i=0; i<m_2; ++i)
	{
		/* init for code beauty :) */
		traces[i]=0;

		/* p and p */
		traces[i]+=m_kernel->kernel(i, m_2+i);

		/* q and q */
		traces[i]+=m_kernel->kernel(m+i, m+m_2+i);

		/* p and q */
		traces[i]-=m_kernel->kernel(i, m+m_2+i);

		/* q and p */
		traces[i]-=m_kernel->kernel(m_2+i, m+i);
	}

	/* return linear time variance estimate */
	return CStatistics::variance(traces)/m_2;
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

