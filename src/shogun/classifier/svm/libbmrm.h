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

#ifndef libbmrm_h
#define libbmrm_h

#define LIBBMRM_PLUS_INF (-log(0.0))
#define LIBBMRM_CALLOC(x, y) calloc(x, y)
#define LIBBMRM_FREE(x) SG_FREE(x)
#define LIBBMRM_MEMCPY(x, y, z) memcpy(x, y, z)
#define LIBBMRM_INDEX(ROW, COL, NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))
#define LIBBMRM_ABS(A) ((A) < 0 ? -(A) : (A))

namespace shogun
{
typedef struct {
    uint32_t nIter;         /* number of iterations  */
    uint32_t nCP;           /* number of cutting planes */
    float64_t Fp;           /* primal objective value  */
    float64_t Fd;           /* reduced (dual) objective value */
    int8_t qp_exitflag;     /* exitflag from the last call of the inner QP solver  */
    int8_t exitflag;        /* 1 .. ocas.Q_P - ocas.Q_D <= TolRel*ABS(ocas.Q_P)
                               2 .. ocas.Q_P - ocas.Q_D <= TolAbs
                              -1 .. ocas.nCutPlanes >= BufSize
                              -2 .. not enough memory for the solver */
} bmrm_return_value_T;

/* standard BMRM solver */
bmrm_return_value_T svm_bmrm_solver(
    void *data,
    float64_t *W,
    float64_t TolRel,
    float64_t TolAbs,
    float64_t lambda,
    uint32_t _BufSize,
    uint32_t (*get_dim)(void*),
    void (*risk_function)(void*, float64_t*, float64_t*, float64_t*)
    );
}

#endif /* libbmrm_h */
