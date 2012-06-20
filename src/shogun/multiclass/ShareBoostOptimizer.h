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

#include <shogun/multiclass/ShareBoost.h>

namespace shogun
{

class ShareBoostOptimizer
{
public:
	/** constructor */
	ShareBoostOptimizer(CShareBoost *sb)
		:m_sb(sb) { SG_REF(m_sb); }
	/** destructor */
	~ShareBoostOptimizer() { SG_UNREF(m_sb); }

	/** run optimization to compute the coefficients */
	void optimize();
private:
	/** the callback for l-bfgs */
	static float64_t lbfgs_evaluate(void *userdata, float64_t *W, float64_t *grad, const int32_t n, const float64_t step);

	CShareBoost *m_sb;
};

} /* shogun */ 

#endif /* end of include guard: SHAREBOOSTOPTIMIZER_H__ */

