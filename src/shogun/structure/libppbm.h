/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * libppbm.h: Implementation of the Proximal Point BM solver for SO training
 *
 * Copyright (C) 2012 Michal Uricar, uricamic@cmp.felk.cvut.cz
 *
 * Implementation of the Proximal Point Bundle Method solver
 *--------------------------------------------------------------------- */

#include <shogun/lib/common.h>
#include <shogun/structure/RiskFunction.h>
#include <shogun/structure/libbmrm.h>

#ifndef libppbm_h
#define libppbm_h

namespace shogun
{
	/* Bundle Methods Solver for Structured Output Learning */
    bmrm_return_value_T svm_ppbm_solver(
			bmrm_data_T 	*data,
			float64_t   	*W,
			float64_t   	TolRel,
			float64_t   	TolAbs,
			float64_t   	lambda,
			uint32_t    	_BufSize,
			bool        	cleanICP,
			uint32_t    	cleanAfter,
			float64_t   	K,
			uint32_t    	Tmax,
			bool        	verbose,
			CRiskFunction* 	risk_function
			);

}

#endif /* libppbm_h */
