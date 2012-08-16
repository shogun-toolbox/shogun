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
#include <shogun/structure/libbmrm.h>

#ifndef libppbm_h
#define libppbm_h

namespace shogun
{
	/** Proximal Point BMRM Solver for Structured Output Learning
	 *
	 * @param model			Pointer to user defined CStructuredModel
	 * @param W				Weight vector
	 * @param TolRel		Relative tolerance
	 * @param TolAbs		Absolute tolerance
	 * @param _lambda		Regularization constant
	 * @param _BufSize		Size of the CP buffer (i.e. maximal number of iterations)
	 * @param cleanICP		Flag that enables/disables inactive cutting plane removal
	 * 						feature
	 * @param cleanAfter	Number of iterations that should be cutting plane
	 * 						inactive for to be removed
	 * @param K				Parameter K
	 * @param Tmax			Parameter Tmax
	 * @param verbose		Flag that enables/disables screen output
	 * @return Structure with BMRM algorithm result
	 */
	bmrm_return_value_T svm_ppbm_solver(
			CStructuredModel *model,
			float64_t   	*W,
			float64_t   	TolRel,
			float64_t   	TolAbs,
			float64_t   	_lambda,
			uint32_t    	_BufSize,
			bool        	cleanICP,
			uint32_t    	cleanAfter,
			float64_t   	K,
			uint32_t    	Tmax,
			bool        	verbose
			);

}

#endif /* libppbm_h */
