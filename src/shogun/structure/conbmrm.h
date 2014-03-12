/*
* This class unifies the code in the present convex bundle solvers into a single file.
* TODO : Unify all the convex bundle solvers.
*/
#ifndef CONBMRM_H
#define CONBMRM_H

#include <shogun/lib/common.h>
#include <shogun/structure/BmrmStatistics.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>
#include <shogun/io/SGIO.h>

#define LIBBMRM_PLUS_INF (-log(0.0))
#define LIBBMRM_CALLOC(x, y) SG_CALLOC(y, x)
#define LIBBMRM_REALLOC(x, y) SG_REALLOC(x, y)
#define LIBBMRM_FREE(x) SG_FREE(x)
#define LIBBMRM_MEMCPY(x, y, z) memcpy(x, y, z)
#define LIBBMRM_MEMMOVE(x, y, z) memmove(x, y, z)
#define LIBBMRM_INDEX(ROW, COL, NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))
#define LIBBMRM_ABS(A) ((A) < 0 ? -(A) : (A))

namespace shogun
{
extern uint32_t BufSize;
/*********** 1. Common method for all solvers ****************/

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
	//MOD
	void bmrm_ll_init(bmrm_ll* prv, bmrm_ll* nxt, float64_t* adrs, uint32_t i)
	{
		prev = prv;
		next = next;
		address = adrs;
		idx = i;
	}
	//MOD
};


/** inactive cutting plane statistics */
IGNORE_IN_CLASSLIST struct ICP_stats
{
	/** maximum number of CP stats we can hold */
	uint32_t maxCPs;

	/** vector of the number of iterations the CPs were inactive */
	uint32_t* ICPcounter;

	/** vector of addresses of the inactive CPs that needs to be pruned */
	float64_t** ICPs;

	/** vector of the active CPs */
	uint32_t* ACPs;

	/** Temporary buffer for storing H */
	float64_t* H_buff;
};

/** Add cutting plane
 *
 * @param tail		Pointer to the last CP entry
 * @param map		Pointer to map storing info about CP physical memory
 * @param A			CP physical memory
 * @param free_idx	Index to physical memory where the CP data will be stored
 * @param cp_data	CP data
 * @param dim		Dimension of CP data
 */

void add_cutting_plane(
		bmrm_ll**	tail,
		bool*		map,
		float64_t*	A,
		uint32_t	free_idx,
		float64_t*	cp_data,
		uint32_t	dim);

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
		float64_t*	icp);

/**
 * Clean-up in-active cutting planes
 */
void clean_icp(ICP_stats* icp_stats,
		BmrmStatistics& bmrm,
		bmrm_ll** head,
		bmrm_ll** tail,
		float64_t*& H,
		float64_t*& diag_H,
		float64_t*& beta,
		bool*& map,
		uint32_t cleanAfter,
		float64_t*& b,
		uint32_t*& I,
		uint32_t cp_models = 0
		);

/** Get cutting plane
 *
 * @param ptr	Pointer to some CP entry
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
{
    for (uint32_t i=0; i<size; ++i) if (map[i]) return i;
    SG_SERROR("No free index available in CP buffer of size %d.\n", size);
    return size-1;
}

/************** 2. Solver methods *******************/

/**** I.  Standard BMRM Solver for Structured Output Learning *******
 *
 * @param machine		Pointer to the BMRM machine
 * @param W				Weight vector
 * @param TolRel		Relative tolerance
 * @param TolAbs		Absolute tolerance
 * @param _lambda		Regularization constant
 * @param _BufSize		Size of the CP buffer (i.e. maximal number of
 *						iterations)
 * @param cleanICP		Flag that enables/disables inactive cutting plane
 *						removal feature
 * @param cleanAfter	Number of iterations that should be cutting plane
 *						inactive for to be removed
 * @param K				Parameter K
 * @param Tmax			Parameter Tmax
 * @param verbose		Flag that enables/disables screen output
 * @param solvertype	The type of solver (i.e. BMRM, PPBM or P3BM)
 * @return Structure with BMRM algorithm result
 */
BmrmStatistics com_bmrm_solver(
		CDualLibQPBMSOSVM  *machine,
		float64_t          *W,
		float64_t          TolRel,
		float64_t          TolAbs,
		float64_t          _lambda,
		uint32_t           _BufSize,
		bool               cleanICP,
		uint32_t           cleanAfter,
		float64_t          K,
		uint32_t           Tmax,
		uint32_t			cp_models,
		bool               verbose,
		uint32_t			solvertype
		);
}

#endif /*UNIBMRM_H*/

