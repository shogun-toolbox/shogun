/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Yuyu Zhang, Bjoern Esser
 */

#ifndef SHAREBOOSTOPTIMIZER_H__
#define SHAREBOOSTOPTIMIZER_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/ShareBoost.h>

namespace shogun
{

/** Utility for ShareBoost to handle optimization */
class ShareBoostOptimizer
{
public:
	/** constructor */
	ShareBoostOptimizer(std::shared_ptr<ShareBoost >sb, bool verbose=false)
		:m_sb(sb), m_verbose(verbose) {  }
	/** destructor */
	~ShareBoostOptimizer() {  }

	/** run optimization to compute the coefficients */
	void optimize();
private:
	/** the callback for l-bfgs */
	static float64_t lbfgs_evaluate(void *userdata, const float64_t *W, float64_t *grad, const int32_t n, const float64_t step);

	/** the callback for logging */
	static int lbfgs_progress(
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
			);

	std::shared_ptr<ShareBoost >m_sb;
	bool m_verbose;
};

} /* shogun */

#endif /* end of include guard: SHAREBOOSTOPTIMIZER_H__ */

