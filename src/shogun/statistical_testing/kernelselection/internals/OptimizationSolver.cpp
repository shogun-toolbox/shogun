/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2014 - 2017 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <functional>
#include <algorithm>
#include <numeric>
#include <vector>
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/statistical_testing/kernelselection/internals/OptimizationSolver.h>

//#ifdef USE_GPL_SHOGUN
#include <shogun/lib/external/libqp.h>
//#endif // USE_GPL_SHOGUN

using namespace shogun;
using namespace internal;

struct OptimizationSolver::Self
{
	Self(SGVector<float64_t> mmds, SGMatrix<float64_t> Q);
//#ifdef USE_GPL_SHOGUN
	SGVector<float64_t> solve() const;
	void init();
	static const float64_t* get_Q_col(uint32_t i);
	static void print_state(libqp_state_T state);

	index_t opt_max_iterations;
	float64_t opt_epsilon;
	float64_t opt_low_cut;
	SGVector<float64_t> m_mmds;
	static SGMatrix<float64_t> m_Q;
//#endif // USE_GPL_SHOGUN
};

//#ifdef USE_GPL_SHOGUN
SGMatrix<float64_t> OptimizationSolver::Self::m_Q=SGMatrix<float64_t>();
//#endif // USE_GPL_SHOGUN

OptimizationSolver::Self::Self(SGVector<float64_t> mmds, SGMatrix<float64_t> Q)
{
//#ifdef USE_GPL_SHOGUN
	m_Q=Q;
	m_mmds=mmds;
	init();
//#endif // USE_GPL_SHOGUN
}

//#ifdef USE_GPL_SHOGUN
void OptimizationSolver::Self::init()
{
	opt_max_iterations=10000;
	opt_epsilon=1E-14;
	opt_low_cut=1E-6;
}

const float64_t* OptimizationSolver::Self::get_Q_col(uint32_t i)
{
	return &m_Q[m_Q.num_rows*i];
}

void OptimizationSolver::Self::print_state(libqp_state_T state)
{
	SG_SDEBUG("libqp state: primal=%f\n", state.QP);
}

SGVector<float64_t> OptimizationSolver::Self::solve() const
{
	const index_t num_kernels=m_mmds.size();
	float64_t sum_m_mmds=std::accumulate(m_mmds.data(), m_mmds.data()+m_mmds.size(), 0);
	SGVector<float64_t> weights(num_kernels);
	if (std::any_of(m_mmds.data(), m_mmds.data()+m_mmds.size(), [](float64_t& value) { return value > 0; }))
	{
		SG_SDEBUG("At least one MMD entry is positive, performing optimisation\n")

		std::vector<float64_t> Q_diag(num_kernels);
		std::vector<float64_t> f(num_kernels, 0);
		std::vector<float64_t> lb(num_kernels, 0);
		std::vector<float64_t> ub(num_kernels, CMath::INFTY);

		// initial point has to be feasible, i.e. m_mmds'*x = b
		std::fill(weights.data(), weights.data()+weights.size(), 1.0/sum_m_mmds);

		for (index_t i=0; i<num_kernels; ++i)
			Q_diag[i]=m_Q(i,i);

		SG_SDEBUG("starting libqp optimization\n");
		libqp_state_T qp_exitflag=libqp_gsmo_solver(&OptimizationSolver::Self::get_Q_col,
			Q_diag.data(),
			f.data(),
			m_mmds.data(),
			1,
			lb.data(),
			ub.data(),
			weights.data(),
			num_kernels,
			opt_max_iterations,
			opt_epsilon,
			&OptimizationSolver::Self::print_state);

		SG_SDEBUG("libqp returns: nIts=%d, exit_flag: %d\n", qp_exitflag.nIter, qp_exitflag.exitflag);
		m_Q=SGMatrix<float64_t>();

		// set really small entries to zero and sum up for normalization
		float64_t sum_weights=0;
		for (index_t i=0; i<weights.vlen; ++i)
		{
			if (weights[i]<opt_low_cut)
			{
				SG_SDEBUG("lowcut: weight[%i]=%f<%f setting to zero\n", i, weights[i], opt_low_cut);
				weights[i]=0;
			}
			sum_weights+=weights[i];
		}

		// normalize (allowed since problem is scale invariant)
		std::for_each(weights.data(), weights.data()+weights.size(), [&sum_weights](float64_t& weight)
		{
			weight/=sum_weights;
		});
	}
	else
	{
		SG_SWARNING("All mmd estimates are negative. This is techically possible,"
			"although extremely rare. Consider using different kernels. "
			"This combination will lead to a bad two-sample test. Since any"
			"combination is bad, will now just return equally distributed "
			"kernel weights\n");

		// if no element is positive, we can choose arbritary weights since
		// the results will be bad anyway
		std::fill(weights.data(), weights.data()+weights.size(), 1.0/num_kernels);
	}
	return weights;
}
//#endif // USE_GPL_SHOGUN

OptimizationSolver::OptimizationSolver(const SGVector<float64_t>& mmds, const SGMatrix<float64_t>& Q)
{
	self=std::unique_ptr<Self>(new Self(mmds, Q));
}

OptimizationSolver::~OptimizationSolver()
{
}

SGVector<float64_t> OptimizationSolver::solve() const
{
//#ifdef USE_GPL_SHOGUN
	return self->solve();
//#else // USE_GPL_SHOGUN
//	SG_SWARNING("Presently this feature is only available with GNU GPLv3 license!");
//	return SGVector<float64_t>();
//#endif // USE_GPL_SHOGUN
}
