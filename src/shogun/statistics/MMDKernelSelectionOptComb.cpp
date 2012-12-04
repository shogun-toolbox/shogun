/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/MMDKernelSelectionOptComb.h>
#include <shogun/statistics/LinearTimeMMD.h>


using namespace shogun;

CMMDKernelSelectionOptComb::CMMDKernelSelectionOptComb() :
		CMMDKernelSelection()
{
	init();
}

CMMDKernelSelectionOptComb::CMMDKernelSelectionOptComb(
		CKernelTwoSampleTestStatistic* mmd, float64_t lambda) :
		CMMDKernelSelection(mmd)
{
	init();

	/* currently, this method is only developed for the linear time MMD */
	REQUIRE(dynamic_cast<CLinearTimeMMD*>(mmd), "%s::%s(): Only "
			"CLinearTimeMMD is currently supported! Provided instance is "
			"\"%s\"\n", get_name(), get_name(), mmd->get_name());

	m_lambda=lambda;
}

CMMDKernelSelectionOptComb::~CMMDKernelSelectionOptComb()
{
}

float64_t CMMDKernelSelectionOptComb::compute_measure(CKernel* kernel)
{
	/* we do not select the kernel via selecting the best of single measures */
	SG_ERROR("%s::compute_measure(): Method must not be used, use"
			" select_kernel() instead!\n", get_name());
	return 0;
}

SGVector<float64_t> CMMDKernelSelectionOptComb::compute_measures()
{
	/* we do not select the kernel via selecting the best of single measures */
	SG_ERROR("%s::compute_measure(): Method must not be used, use"
			" select_kernel() instead!\n", get_name());
	return SGVector<float64_t>();
}

void CMMDKernelSelectionOptComb::init()
{
	/* set to a sensible standard value that proved to be useful in
	 * experiments */
	m_lambda=10E-5;
}

#ifdef HAVE_LAPACK
CKernel* CMMDKernelSelectionOptComb::select_kernel()
{
//	/* use MaxMMD class to compute MMD for all underlying kernels */
//	CMMDKernelSelectionMax* maxmmd=new CMMDKernelSelectionMax(m_mmd);
//
//
//	/* think about casting and types of different kernels/features here */
//	CCombinedFeatures* combined_p_and_q=
//			dynamic_cast<CCombinedFeatures*>(m_p_and_q);
//	CCombinedKernel* combined_kernel=dynamic_cast<CCombinedKernel*>(m_kernel);
//	ASSERT(combined_p_and_q);
//	ASSERT(combined_kernel);
//
//	if (combined_kernel->get_num_subkernels()!=
//			combined_p_and_q->get_num_feature_obj())
//	{
//		SG_ERROR("CLinearTimeMMD::optimize_kernel_weights(): Only possible "
//				"when number of sub-kernels (%d) equal number of sub-features "
//				"(%d)\n", combined_kernel->get_num_subkernels(),
//				combined_p_and_q->get_num_feature_obj());
//	}
//
//	/* init kernel with features */
//	m_kernel->init(m_p_and_q, m_p_and_q);
//
//	/* number of kernels and data */
//	index_t num_kernels=combined_kernel->get_num_subkernels();
//	index_t m2=m_m/2;
//
//	/* matrix with all h entries for all kernels and data */
//	SGMatrix<float64_t> hs(m2, num_kernels);
//
//	/* mmds are needed and are means of columns of hs */
//	SGVector<float64_t> mmds(num_kernels);
//
//	float64_t pp;
//	float64_t qq;
//	float64_t pq;
//	float64_t qp;
//	/* compute all h entries */
//	for (index_t i=0; i<num_kernels; ++i)
//	{
//		CKernel* current=combined_kernel->get_kernel(i);
//		mmds[i]=0;
//		for (index_t j=0; j<m2; ++j)
//		{
//			pp=current->kernel(j, m2+j);
//			qq=current->kernel(m_m+j, m_m+m2+j);
//			pq=current->kernel(j, m_m+m2+j);
//			qp=current->kernel(m2+j, m_m+j);
//			hs(j, i)=pp+qq-pq-qp;
//			mmds[i]+=hs(j, i);
//		}
//
//		/* mmd is simply mean. This is the unbiased linear time estimate */
//		mmds[i]/=m2;
//
//		SG_UNREF(current);
//	}
//
//	/* compute covariance matrix of h vector, in place is safe now since h
//	 * is not needed anymore */
//	m_Q=CStatistics::covariance_matrix(hs, true);
//
//	/* evtl regularize to avoid numerical problems (ratio of MMD and std-dev
//	 * blows up when variance is small */
//	if (m_opt_regularization_eps)
//	{
//		SG_DEBUG("regularizing matrix Q by adding %f to diagonal\n",
//				m_opt_regularization_eps);
//		for (index_t i=0; i<num_kernels; ++i)
//			m_Q(i,i)+=m_opt_regularization_eps;
//	}
//
//	if (sg_io->get_loglevel()==MSG_DEBUG)
//	{
//		m_Q.display_matrix("(evtl. regularized) Q");
//		mmds.display_vector("mmds");
//	}
//
//	/* compute sum of mmds to generate feasible point for convex program */
//	float64_t sum_mmds=0;
//	for (index_t i=0; i<mmds.vlen; ++i)
//		sum_mmds+=mmds[i];
//
//	/* QP: 0.5*x'*Q*x + f'*x
//	 * subject to
//	 * mmds'*x = b
//	 * LB[i] <= x[i] <= UB[i]   for all i=1..n */
//	SGVector<float64_t> Q_diag(num_kernels);
//	SGVector<float64_t> f(num_kernels);
//	SGVector<float64_t> lb(num_kernels);
//	SGVector<float64_t> ub(num_kernels);
//	SGVector<float64_t> x(num_kernels);
//
//	/* init everything, there are two cases possible: i) at least one mmd is
//	 * is positive, ii) all mmds are negative */
//	bool one_pos;
//	for (index_t i=0; i<mmds.vlen; ++i)
//	{
//		if (mmds[i]>0)
//		{
//			SG_DEBUG("found at least one positive MMD\n");
//			one_pos=true;
//			break;
//		}
//		one_pos=false;
//	}
//
//	if (!one_pos)
//	{
//		SG_WARNING("All mmd estimates are negative. This is techical possible,"
//				" although extremely rare. Current problem might bad\n");
//
//		/* if no element is positive, Q has to be replaced by -Q */
//		for (index_t i=0; i<num_kernels*num_kernels; ++i)
//			m_Q.matrix[i]=-m_Q.matrix[i];
//	}
//
//	/* init vectors */
//	for (index_t i=0; i<num_kernels; ++i)
//	{
//		Q_diag[i]=m_Q(i,i);
//		f[i]=0;
//		lb[i]=0;
//		ub[i]=CMath::INFTY;
//
//		/* initial point has to be feasible, i.e. mmds'*x = b */
//		x[i]=1.0/sum_mmds;
//	}
//
//	/* start libqp solver with desired parameters */
//	SG_DEBUG("starting libqp optimization\n");
//	libqp_state_T qp_exitflag=libqp_gsmo_solver(&get_Q_col, Q_diag.vector,
//			f.vector, mmds.vector,
//			one_pos ? 1 : -1,
//			lb.vector, ub.vector,
//			x.vector, num_kernels, m_opt_max_iterations,
//			m_opt_regularization_eps, &print_state);
//
//	SG_DEBUG("libqp returns: nIts=%d, exit_flag: %d\n", qp_exitflag.nIter,
//			qp_exitflag.exitflag);
//
//	/* set really small entries to zero and sum up for normalization */
//	float64_t sum_weights=0;
//	for (index_t i=0; i<x.vlen; ++i)
//	{
//		if (x[i]<m_opt_low_cut)
//		{
//			SG_DEBUG("lowcut: weight[%i]=%f<%f; setting to zero\n", i, x[i],
//					m_opt_low_cut);
//			x[i]=0;
//		}
//
//		sum_weights+=x[i];
//	}
//
//	/* normalize (allowed since problem is scale invariant) */
//	for (index_t i=0; i<x.vlen; ++i)
//		x[i]/=sum_weights;
//
//	/* set weights to kernel */
//	m_kernel->set_subkernel_weights(x);
	return NULL;
}
#else
CKernel* CMMDKernelSelectionOptComb::select_kernel()
{
	SG_ERROR("%s::select_kernel(): LAPACK needs to be installed in order to use"
			" optimal weight selection for combined kernels!\n", get_name());
	return NULL;
}

SGMatrix<float64_t> CMMDKernelSelectionOptComb::m_Q=SGMatrix<float64_t>();

const float64_t* CMMDKernelSelectionOptComb::get_Q_col(uint32_t i)
{
	return &m_Q[m_Q.num_rows*i];
}

void CMMDKernelSelectionOptComb::print_state(libqp_state_T state)
{
	SG_SDEBUG("%s::print_state: libqp state: primal=%f\n", get_name(),
			state.QP);
}
#endif
