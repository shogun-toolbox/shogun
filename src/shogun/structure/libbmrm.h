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
#include <shogun/structure/StructuredModel.h>

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
#define IGNORE_IN_CLASSLIST

namespace shogun
{
/** BMRM result structure */
//IGNORE_IN_CLASSLIST struct bmrm_return_value_T
struct bmrm_return_value_T
{
	/** number of iterations  */
	uint32_t nIter;

	/** number of cutting planes */
	uint32_t nCP;

	/** number of active cutting planes */
	uint32_t nzA;

	/** primal objective value  */
	float64_t Fp;

	/** reduced (dual) objective value */
	float64_t Fd;

	/** exitflag from the last call of the inner QP solver  */
	int8_t qp_exitflag;

	/** 1 .. bmrm.Q_P - bmrm.Q_D <= TolRel*ABS(bmrm.Q_P)
	 *  2 .. bmrm.Q_P - bmrm.Q_D <= TolAbs
	 * -1 .. bmrm.nCutPlanes >= BufSize
	 * -2 .. not enough memory for the solver
	 */
	int8_t exitflag;

	/** Track of Fp values in individual iterations */
	SGVector< float64_t > hist_Fp;

	/** Track of Fd values in individual iterations */
	SGVector< float64_t > hist_Fd;

	/** Track of w_dist values in individual iterations */
	SGVector< float64_t > hist_wdist;
};

/** Linked list for cutting planes buffer management */
IGNORE_IN_CLASSLIST struct bmrm_ll {
	/** Pointer to previous CP entry */
	bmrm_ll   *prev;
	/** Pointer to next CP entry */
	bmrm_ll   *next;
	/** Pointer to the real CP data */
	float64_t   *address;
	/** Index of CP */
	uint32_t    idx;
};

/** Add cutting plane
 *
 * @param tail 		Pointer to the last CP entry
 * @param map		Pointer to map storing info about CP physical memory
 * @param A			CP physical memory
 * @param free_idx	Index to physical memory where the CP data will be stored
 * @param cp_data	CP data
 * @param dim		Dimension of CP data
 */
void add_cutting_plane(
		bmrm_ll**	tail,
		bool* 		map,
		float64_t*	A,
		uint32_t 	free_idx,
		float64_t* 	cp_data,
		uint32_t 	dim);

/** Remove cutting plane at given index
 *
 * @param head	Pointer to the first CP entry
 * @param tail	Pointer to the last CP entry
 * @param map	Pointer to map storing info about CP physical memory
 * @param icp	Pointer to inactive CP that should be removed
 */
void remove_cutting_plane(
		bmrm_ll**	head,
		bmrm_ll**	tail,
		bool*		map,
		float64_t* 	icp);

/** Get cutting plane
 *
 * @param ptr 	Pointer to some CP entry
 * @return Pointer to cutting plane at given entry
 */
inline float64_t * get_cutting_plane(bmrm_ll *ptr) { return ptr->address; }

/** Get index of free slot for new cutting plane
 *
 * @param map	Pointer to map storing info about CP physical memory
 * @param size	Size of the CP buffer
 * @return Index of unoccupied memory field in CP physical memory
 */
inline uint32_t find_free_idx(bool *map, uint32_t size)
{ for(uint32_t i=0; i<size; ++i) if (map[i]) return i; return size+1; }

/** Standard BMRM Solver for Structured Output Learning
 *
 * @param model			Pointer to user defined CStructuredModel
 * @param W				Weight vector
 * @param TolRel		Relative tolerance
 * @param TolAbs		Absolute tolerance
 * @param _lambda		Regularization constant
 * @param _BufSize		Size of the CP buffer (i.e. maximal number of
 * 						iterations)
 * @param cleanICP		Flag that enables/disables inactive cutting plane
 * 						removal feature
 * @param cleanAfter	Number of iterations that should be cutting plane
 * 						inactive for to be removed
 * @param K				Parameter K
 * @param Tmax			Parameter Tmax
 * @param verbose		Flag that enables/disables screen output
 * @return Structure with BMRM algorithm result
 */
bmrm_return_value_T svm_bmrm_solver(
		CStructuredModel *model,
		float64_t        *W,
		float64_t        TolRel,
		float64_t        TolAbs,
		float64_t        _lambda,
		uint32_t         _BufSize,
		bool             cleanICP,
		uint32_t         cleanAfter,
		float64_t        K,
		uint32_t         Tmax,
		bool             verbose
		);

}

#endif /* libbmrm_h */
