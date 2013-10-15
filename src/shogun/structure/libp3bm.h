/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * libp3bm.h: Implementation of the Proximal Point P-BMRM solver for SO training
 *
 * Copyright (C) 2012 Michal Uricar, uricamic@cmp.felk.cvut.cz
 *
 * Implementation of the Proximal Point P-BMRM (3pbm)
 *--------------------------------------------------------------------- */

#ifndef libp3bm_h
#define libp3bm_h

#include <shogun/lib/common.h>
#include <shogun/structure/libbmrm.h>

namespace shogun
{
	/** Proximal Point P-BMRM (multiple cutting plane models) Solver for
	 *	Structured Output Learning
	 *
	 * @param machine		Pointer to the BMRM machine
	 * @param W				Weight vector
	 * @param TolRel		Relative tolerance
	 * @param TolAbs		Absolute tolerance
	 * @param _lambda		Regularization constant
	 * @param _BufSize		Size of the CP buffer (i.e. maximal number of iterations)
	 * @param cleanICP		Flag that enables/disables inactive cutting plane removal
	 *						feature
	 * @param cleanAfter	Number of iterations that should be cutting plane
	 *						inactive for to be removed
	 * @param K				Parameter K
	 * @param Tmax			Parameter Tmax
	 * @param cp_models		Count of cutting plane models to be used
	 * @param verbose		Flag that enables/disables screen output
	 * @return Structure with BMRM algorithm result
	 */
	BmrmStatistics svm_p3bm_solver(
			CDualLibQPBMSOSVM  *machine,
			float64_t	*W,
			float64_t	TolRel,
			float64_t	TolAbs,
			float64_t	_lambda,
			uint32_t	_BufSize,
			bool	cleanICP,
			uint32_t	cleanAfter,
			float64_t	K,
			uint32_t	Tmax,
			uint32_t        cp_models,
			bool	verbose
			);

}

#endif /* libp3bm_h */
