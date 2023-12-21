/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg, Bjoern Esser
 */

#include <algorithm>

#include <shogun/mathematics/Math.h>
#include <shogun/optimization/lbfgs/lbfgs.h>
#include <shogun/multiclass/ShareBoostOptimizer.h>

using namespace shogun;

void ShareBoostOptimizer::optimize()
{
	int32_t N = m_sb->m_multiclass_strategy->get_num_classes() * m_sb->m_activeset.vlen;
	float64_t *W = SG_CALLOC(float64_t, N); // should use this function, if sse is enabled for liblbfgs
	float64_t objval;
	lbfgs_parameter_t param;
	lbfgs_parameter_init(&param);

	std::fill(W, W+N, 0);

	lbfgs_progress_t progress = m_verbose ? &ShareBoostOptimizer::lbfgs_progress : NULL;

	lbfgs(N, W, &objval, &ShareBoostOptimizer::lbfgs_evaluate, progress, this, &param);

	int32_t w_len = m_sb->m_activeset.vlen;
	for (int32_t i=0; i < m_sb->m_multiclass_strategy->get_num_classes(); ++i)
	{
		auto machine = m_sb->m_machines.at(i)->as<LinearMachine>();
		SGVector<float64_t> w(w_len);
		std::copy(W + i*w_len, W + (i+1)*w_len, w.vector);
		machine->set_w(w);

	}

	SG_FREE(W);
}

float64_t ShareBoostOptimizer::lbfgs_evaluate(void *userdata, const float64_t *W,
		float64_t *grad, const int32_t n, const float64_t step)
{
	ShareBoostOptimizer *optimizer = static_cast<ShareBoostOptimizer *>(userdata);

	optimizer->m_sb->compute_pred(W);
	optimizer->m_sb->compute_rho(optimizer->m_sb->m_share_boost_labels);

	int32_t m = optimizer->m_sb->m_activeset.vlen;
	int32_t k = optimizer->m_sb->m_multiclass_strategy->get_num_classes();

	SGMatrix<float64_t> fea = optimizer->m_sb->m_fea;
	auto lab = multiclass_labels(optimizer->m_sb->m_share_boost_labels);

	// compute gradient
	for (int32_t i=0; i < m; ++i)
	{
		for (int32_t j=0; j < k; ++j)
		{
			int32_t idx = j*m + i;
			float64_t g=0;
			for (int32_t ii=0; ii < fea.num_cols; ++ii)
				g += fea(optimizer->m_sb->m_activeset[i], ii) *
					(optimizer->m_sb->m_rho(j,ii)/optimizer->m_sb->m_rho_norm[ii] -
					 (j == lab->get_int_label(ii)));
			g /= fea.num_cols;
			grad[idx] = g;
		}
	}

	// compute objective function
	float64_t objval = 0;
	for (int32_t ii=0; ii < fea.num_cols; ++ii)
	{
		objval += std::log(optimizer->m_sb->m_rho_norm[ii]);
	}
	objval /= fea.num_cols;

	return objval;
}

int ShareBoostOptimizer::lbfgs_progress(
		void *instance,
		const float64_t *x,
		const float64_t *g,
		const float64_t fx,
		const float64_t xnorm,
		const float64_t gnorm,
		const float64_t step,
		int n,
		int k,
		int ls
		)
{
	if (k != 1 && k % 100 != 0)
		return 0;

    io::print("Iteration {}:\n", k);
    io::print("  fx = {}, x[0] = {}, x[1] = {}\n", fx, x[0], x[1]);
    io::print("  xnorm = {}, gnorm = {}, step = {}\n", xnorm, gnorm, step);
    io::print("\n");
    return 0;
}
