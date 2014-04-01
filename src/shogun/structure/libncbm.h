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

#include <shogun/lib/config.h>

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
	BmrmStatistics svm_ncbm_solver(
			CDualLibQPBMSOSVM  *machine,
			float64_t        *w,
			float64_t        TolRel,
			float64_t        TolAbs,
			float64_t        _lambda,
			uint32_t         _BufSize,
			bool             cleanICP,
			uint32_t         cleanAfter,
			bool             is_convex = false,
			bool             line_search = true,
			bool             verbose = false
			);
}
#endif /* libncbm_h */
