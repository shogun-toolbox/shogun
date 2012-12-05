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
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/mathematics/Statistics.h>


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
SGMatrix<float64_t> CMMDKernelSelectionOptComb::m_Q=SGMatrix<float64_t>();

const float64_t* CMMDKernelSelectionOptComb::get_Q_col(uint32_t i)
{
	return &m_Q[m_Q.num_rows*i];
}

void CMMDKernelSelectionOptComb::print_state(libqp_state_T state)
{
	SG_SDEBUG("CMMDKernelSelectionOptComb::print_state: libqp state:"
			" primal=%f\n", state.QP);
}

CKernel* CMMDKernelSelectionOptComb::select_kernel()
{
	/* for readability */
	index_t num_kernels=m_kernel_list->get_num_elements();

	/* result kernel is a combined one */
	CCombinedKernel* combined_kernel=new CCombinedKernel();
	SG_REF(combined_kernel);

	/* go through all kernels and compute h-terms. Copy them to a matrix H */
	SGMatrix<float64_t> H(m_mmd->get_m()/2, num_kernels);
	CKernel* current=(CKernel*)m_kernel_list->get_first_element();
	index_t count=0;
	while (current)
	{
		/* construct combined kernel on the fly */
		combined_kernel->append_kernel(current);

		/* compute h-terms for current kernel */
		m_mmd->set_kernel(current);
		SGVector<float64_t> h=((CLinearTimeMMD*)m_mmd)->compute_h_terms();

		/* copy vector wise to H matrix */
		memcpy(&H(0, count), h.vector, h.vlen*sizeof(float64_t));

		/* proceed to next kernel */
		SG_UNREF(current);
		current=(CKernel*)m_kernel_list->get_next_element();
		++count;
	}

	/* compute mean values (=MMDs) */
	SGVector<float64_t> mmds=CStatistics::matrix_mean(H);

	/* compute empirical covariance matrix Q from H, H can be overwritten */
	m_Q=CStatistics::covariance_matrix(H, true);

	/* evtl regularize to avoid numerical problems (see NIPS paper) */
	if (m_lambda)
	{
		SG_DEBUG("regularizing matrix Q by adding %f to diagonal\n", m_lambda);
		for (index_t i=0; i<num_kernels; ++i)
			m_Q(i,i)+=m_lambda;
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
				" although extremely rare. Consider using different kernels\n");

		/* if no element is positive, we can choose arbritary weights since
		 * the results will be bad anyway */
		SG_NOTIMPLEMENTED;
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
			m_opt_epsilon, &print_state);

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

	/* set combined kernel weights and return */
	combined_kernel->set_subkernel_weights(x);

	SG_WARNING("CMMDKernelSelectionOptComb::select_kernel() is not tested!\n");

	return combined_kernel;
}
#else
CKernel* CMMDKernelSelectionOptComb::select_kernel()
{
	SG_ERROR("%s::select_kernel(): LAPACK needs to be installed in order to use"
			" optimal weight selection for combined kernels!\n", get_name());
	return NULL;
}

}
#endif
