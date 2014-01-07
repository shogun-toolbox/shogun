/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <statistics/MMDKernelSelectionComb.h>
#include <statistics/KernelTwoSampleTestStatistic.h>
#include <kernel/CombinedKernel.h>

using namespace shogun;

CMMDKernelSelectionComb::CMMDKernelSelectionComb() :
		CMMDKernelSelection()
{
	init();
}

CMMDKernelSelectionComb::CMMDKernelSelectionComb(
		CKernelTwoSampleTestStatistic* mmd) : CMMDKernelSelection(mmd)
{
	init();
}

CMMDKernelSelectionComb::~CMMDKernelSelectionComb()
{
}

void CMMDKernelSelectionComb::init()
{
#ifdef HAVE_LAPACK
	SG_ADD(&m_opt_max_iterations, "opt_max_iterations", "Maximum number of "
			"iterations for qp solver", MS_NOT_AVAILABLE);
	SG_ADD(&m_opt_epsilon, "opt_epsilon", "Stopping criterion for qp solver",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_opt_low_cut, "opt_low_cut", "Low cut value for optimization "
			"kernel weights", MS_NOT_AVAILABLE);

	/* sensible values for optimization */
	m_opt_max_iterations=10000;
	m_opt_epsilon=10E-15;
	m_opt_low_cut=10E-7;
#endif
}

#ifdef HAVE_LAPACK
/* no reference counting, use the static context constructor of SGMatrix */
SGMatrix<float64_t> CMMDKernelSelectionComb::m_Q=SGMatrix<float64_t>(false);

const float64_t* CMMDKernelSelectionComb::get_Q_col(uint32_t i)
{
	return &m_Q[m_Q.num_rows*i];
}

/** helper function that prints current state */
void CMMDKernelSelectionComb::print_state(libqp_state_T state)
{
	SG_SDEBUG("CMMDKernelSelectionComb::print_state: libqp state:"
			" primal=%f\n", state.QP);
}

CKernel* CMMDKernelSelectionComb::select_kernel()
{
	/* cast is safe due to assertion in constructor */
	CCombinedKernel* combined=(CCombinedKernel*)m_mmd->get_kernel();

	/* optimise for kernel weights and set them */
	SGVector<float64_t> weights=compute_measures();
	combined->set_subkernel_weights(weights);

	/* note that kernel is SG_REF'ed from getter above */
	return combined;
}

SGVector<float64_t> CMMDKernelSelectionComb::solve_optimization(
		SGVector<float64_t> mmds)
{
	/* readability */
	index_t num_kernels=mmds.vlen;

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
	SGVector<float64_t> weights(num_kernels);

	/* init everything, there are two cases possible: i) at least one mmd is
	 * is positive, ii) all mmds are negative */
	bool one_pos=false;
	for (index_t i=0; i<mmds.vlen; ++i)
	{
		if (mmds[i]>0)
		{
			SG_DEBUG("found at least one positive MMD\n")
			one_pos=true;
			break;
		}
	}

	if (!one_pos)
	{
		SG_WARNING("CMMDKernelSelectionComb::solve_optimization(): all mmd "
				"estimates are negative. This is techically possible, although "
				"extremely rare. Consider using different kernels. "
				"This combination will lead to a bad two-sample test. Since any"
				"combination is bad, will now just return equally distributed "
				"kernel weights\n");

		/* if no element is positive, we can choose arbritary weights since
		 * the results will be bad anyway */
		weights.set_const(1.0/num_kernels);
	}
	else
	{
		SG_DEBUG("one MMD entry is positive, performing optimisation\n")
		/* do optimisation, init vectors */
		for (index_t i=0; i<num_kernels; ++i)
		{
			Q_diag[i]=m_Q(i,i);
			f[i]=0;
			lb[i]=0;
			ub[i]=CMath::INFTY;

			/* initial point has to be feasible, i.e. mmds'*x = b */
			weights[i]=1.0/sum_mmds;
		}

		/* start libqp solver with desired parameters */
		SG_DEBUG("starting libqp optimization\n")
		libqp_state_T qp_exitflag=libqp_gsmo_solver(&get_Q_col, Q_diag.vector,
				f.vector, mmds.vector,
				one_pos ? 1 : -1,
				lb.vector, ub.vector,
				weights.vector, num_kernels, m_opt_max_iterations,
				m_opt_epsilon, &(CMMDKernelSelectionComb::print_state));

		SG_DEBUG("libqp returns: nIts=%d, exit_flag: %d\n", qp_exitflag.nIter,
				qp_exitflag.exitflag);

		/* set really small entries to zero and sum up for normalization */
		float64_t sum_weights=0;
		for (index_t i=0; i<weights.vlen; ++i)
		{
			if (weights[i]<m_opt_low_cut)
			{
				SG_DEBUG("lowcut: weight[%i]=%f<%f setting to zero\n", i, weights[i],
						m_opt_low_cut);
				weights[i]=0;
			}

			sum_weights+=weights[i];
		}

		/* normalize (allowed since problem is scale invariant) */
		for (index_t i=0; i<weights.vlen; ++i)
			weights[i]/=sum_weights;
	}

	return weights;
}
#else
CKernel* CMMDKernelSelectionComb::select_kernel()
{
	SG_ERROR("CMMDKernelSelectionComb::select_kernel(): LAPACK needs to be "
			"installed in order to use weight optimisation for combined "
			"kernels!\n");
	return NULL;
}

SGVector<float64_t> CMMDKernelSelectionComb::compute_measures()
{
	SG_ERROR("CMMDKernelSelectionComb::select_kernel(): LAPACK needs to be "
			"installed in order to use weight optimisation for combined "
			"kernels!\n");
	return SGVector<float64_t>();
}

SGVector<float64_t> CMMDKernelSelectionComb::solve_optimization(
		SGVector<float64_t> mmds)
{
	SG_ERROR("CMMDKernelSelectionComb::solve_optimization(): LAPACK needs to be "
			"installed in order to use weight optimisation for combined "
			"kernels!\n");
	return SGVector<float64_t>();
}
#endif
