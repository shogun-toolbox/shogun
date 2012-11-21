/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Viktor Gal
 *
 */

#ifndef libncbm_h
#define libncbm_h

#include <shogun/lib/common.h>
#include <shogun/structure/libbmrm.h>

namespace shogun
{

	/**
	 * NCBM (non-convex bundle method) solver
	 * Solves any unconstrainedminimization problem in the form of:
	 * min lambda/2 ||w||^2 + R(w)
	 * where R(w) is a risk funciton of any kind.
	 */
	bmrm_return_value_T svm_ncbm_solver(
			CStructuredModel *model,
			float64_t        *w,
			float64_t        TolRel,
			float64_t        TolAbs,
			float64_t        _lambda,
			uint32_t         _BufSize,
			bool             cleanICP,
			uint32_t         cleanAfter,
			bool             is_convex,
			bool             verbose
			);
}
#endif /* libncbm_h */
