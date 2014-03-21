/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
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
	ShareBoostOptimizer(CShareBoost *sb, bool verbose=false)
		:m_sb(sb), m_verbose(verbose) { SG_REF(m_sb); }
	/** destructor */
	~ShareBoostOptimizer() { SG_UNREF(m_sb); }

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

	CShareBoost *m_sb;
	bool m_verbose;
};

} /* shogun */

#endif /* end of include guard: SHAREBOOSTOPTIMIZER_H__ */

