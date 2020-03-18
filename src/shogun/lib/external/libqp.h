/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Vojtech Franc
 */

#ifndef libqp_h
#define libqp_h

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{
#define LIBQP_PLUS_INF (-log(0.0))
#define LIBQP_CALLOC(x,y) SG_CALLOC(y,x)
#define LIBQP_FREE(x) SG_FREE(x)
#define LIBQP_INDEX(ROW,COL,NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))
#define LIBQP_MIN(A,B) ((A) > (B) ? (B) : (A))
#define LIBQP_MAX(A,B) ((A) < (B) ? (B) : (A))
#define LIBQP_ABS(A) ((A) < 0 ? -(A) : (A))

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** QP solver return value */
typedef struct {
  /** number of iterations */
  uint32_t nIter;
  /** primal objective value */
  float64_t QP;
  /** dual objective value */
  float64_t QD;
  /** exit flag */
  int8_t exitflag;      /* -1 ... not enough memory
                            0 ... nIter >= MaxIter
                            1 ... QP - QD <= TolRel*ABS(QP)
                            2 ... QP - QD <= TolAbs
                            3 ... QP <= QP_TH
                            4 ... eps-KKT conditions satisfied */
} libqp_state_T;
#endif

/** QP solver for tasks with simplex constraints */
libqp_state_T libqp_splx_solver(const float64_t* (*get_col)(uint32_t),
                  float64_t *diag_H,
                  float64_t *f,
                  float64_t *b,
                  uint32_t *Ivector,
                  uint8_t *S,
                  float64_t *x,
                  uint32_t n,
                  uint32_t MaxIter,
                  float64_t TolAbs,
                  float64_t TolRel,
                  float64_t QP_TH,
                  void (*print_state)(libqp_state_T state));

/** Generalized SMO algorithm */
libqp_state_T libqp_gsmo_solver(const float64_t* (*get_col)(uint32_t),
            float64_t *diag_H,
            float64_t *f,
            float64_t *a,
            float64_t b,
            float64_t *LB,
            float64_t *UB,
            float64_t *x,
            uint32_t n,
            uint32_t MaxIter,
            float64_t TolKKT,
            void (*print_state)(libqp_state_T state));

}
#endif /* libqp_h */
