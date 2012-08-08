/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * libbmrm.h: Implementation of the BMRM solver for SO training
 *
 * Copyright (C) 2012 Michal Uricar, uricamic@cmp.felk.cvut.cz
 *
 * Implementation of the BMRM solver
 *--------------------------------------------------------------------- */

#include <shogun/lib/common.h>
#include <shogun/structure/RiskFunction.h>

#ifndef libbmrm_h
#define libbmrm_h

#define LIBBMRM_PLUS_INF (-log(0.0))
#define LIBBMRM_CALLOC(x, y) calloc(x, y)
#define LIBBMRM_REALLOC(x, y) realloc(x, y)
#define LIBBMRM_FREE(x) SG_FREE(x)
#define LIBBMRM_MEMCPY(x, y, z) memcpy(x, y, z)
#define LIBBMRM_MEMMOVE(x, y, z) memmove(x, y, z)
#define LIBBMRM_INDEX(ROW, COL, NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))
#define LIBBMRM_ABS(A) ((A) < 0 ? -(A) : (A))

namespace shogun
{
	/* BMRM result */
	typedef struct {
		uint32_t nIter;         /* number of iterations  */
		uint32_t nCP;           /* number of cutting planes */
		uint32_t nzA;           /* number of active cutting planes */
		float64_t Fp;           /* primal objective value  */
		float64_t Fd;           /* reduced (dual) objective value */
		int8_t qp_exitflag;     /* exitflag from the last call of the inner QP solver  */
		int8_t exitflag;        /* 1 .. ocas.Q_P - ocas.Q_D <= TolRel*ABS(ocas.Q_P)
								   2 .. ocas.Q_P - ocas.Q_D <= TolAbs
								   -1 .. ocas.nCutPlanes >= BufSize
								   -2 .. not enough memory for the solver */
	} bmrm_return_value_T;

	/* Linked list for cutting planes buffer management */
	typedef struct ll {
		struct ll   *prev;
		struct ll   *next;
		float64_t   *address;
		uint32_t    idx;
	} bmrm_ll;

	/* Add cutting plane */
	void add_cutting_plane(bmrm_ll **tail, bool *map, float64_t *A, uint32_t free_idx, float64_t *cp_data, uint32_t dim);

	/* Remove cutting plane at given index */
	void remove_cutting_plane(bmrm_ll **head, bmrm_ll **tail, bool *map, float64_t *icp);

	/* Get cutting plane */
	inline float64_t * get_cutting_plane(bmrm_ll *ptr) { return ptr->address; }

	/* Get index of free slot for new cutting plane */
	inline uint32_t find_free_idx(bool *map, uint32_t size) { for(uint32_t i=0; i<size; ++i) if (map[i]) return i; return size+1; }

	/* Bundle Methods Solver for Structured Output Learning */
	bmrm_return_value_T svm_bmrm_solver(
			void        *data,
			float64_t   *W,
			float64_t   TolRel,
			float64_t   TolAbs,
			float64_t   lambda,
			uint32_t    _BufSize,
			bool        cleanICP,
			uint32_t    cleanAfter,
			float64_t   K,
			uint32_t    Tmax,
			bool        verbose,
			CRiskFunction* risk_function
			);

}

#endif /* libbmrm_h */
