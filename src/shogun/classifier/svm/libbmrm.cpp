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

#include <stdlib.h>
#include <string.h>
#include <math.h>

<<<<<<< HEAD
=======
//TEST
#include "libbmrm.h"

>>>>>>> 8f28ad3... BMRM work in progress
#include <shogun/classifier/svm/libbmrm.h>
#include <shogun/classifier/svm/libqp.h>

namespace shogun
{
static const uint32_t QPSolverMaxIter = 10000000;

static float64_t *H;
static uint32_t BufSize;

/*----------------------------------------------------------------------
  Returns pointer at i-th column of Hessian matrix.
  ----------------------------------------------------------------------*/
static const float64_t *get_col( uint32_t i)
{
    return( &H[ BufSize*i ] );
}

bmrm_return_value_T svm_bmrm_solver(
        void *data,
        float64_t *W,
        float64_t TolRel,
        float64_t TolAbs,
        float64_t lambda,
        uint32_t _BufSize,
        uint32_t (*get_dim)(void*),
        void (*risk_function)(void*, float64_t*, float64_t*, float64_t*))
{
<<<<<<< HEAD
    bmrm_return_value_T bmrm = {0, 0, 0, 0, 0, 0};
    libqp_state_T qp_exitflag;
    float64_t *b, *beta, *diag_H, sq_norm_W;
    float64_t R, *subgrad, *A, QPSolverTolRel, rsum, C = 1.0;
=======
    bmrm_return_value bmrm = {0, 0, 0, 0, 0, 0};
    libqp_state_T qp_exitflag;
    float64_t *b, *beta, *diag_H, sq_norm_W;
    float64_t R, *subgrad, *A, QPSolverTolRel, rsum;
>>>>>>> 8f28ad3... BMRM work in progress
    uint32_t nDim;
    uint32_t *I;
    uint8_t S = 1;

    BufSize = _BufSize;

    QPSolverTolRel = TolRel*0.5;

    nDim = get_dim(data);

    H=NULL;
    b=NULL;
    beta=NULL;
    A=NULL;
    subgrad=NULL;
    diag_H = NULL;
    I = NULL;

    LIBBMRM_FREE(W);
    W = NULL;

    H = (float64_t*)LIBBMRM_CALLOC(BufSize*BufSize, sizeof(float64_t));
    if (H == NULL)
    {
        bmrm.exitflag = -2;
        goto cleanup;
    }

    A = (float64_t*)LIBBMRM_CALLOC(nDim*BufSize, sizeof(float64_t));
    if (A == NULL)
    {
        bmrm.exitflag = -2;
        goto cleanup;
    }

    b = (float64_t*)LIBBMRM_CALLOC(BufSize, sizeof(float64_t));
    if (b == NULL)
    {
<<<<<<< HEAD
        bmrm.exitflag = -2;
        goto cleanup;
=======
        bmmr.exitflag = -2;
        goto clenup;
>>>>>>> 8f28ad3... BMRM work in progress
    }

    beta = (float64_t*)LIBBMRM_CALLOC(BufSize, sizeof(float64_t));
    if (beta == NULL)
    {
        bmrm.exitflag = -2;
        goto cleanup;
    }

    subgrad = (float64_t*)LIBBMRM_CALLOC(nDim, sizeof(float64_t));
    if (subgrad == NULL)
    {
        bmrm.exitflag = -2;
        goto cleanup;
    }

    diag_H = (float64_t*)LIBBMRM_CALLOC(BufSize, sizeof(float64_t));
    if (diag_H == NULL)
    {
        bmrm.exitflag = -2;
        goto cleanup;
    }

    I = (uint32_t*)LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));
    if (I == NULL)
    {
        bmrm.exitflag = -2;
        goto cleanup;
    }

    /* initial solution */
    W = (float64_t*)LIBBMRM_CALLOC(nDim, sizeof(float64_t));
    if (W == NULL)
    {
        bmrm.exitflag = -2;
        goto cleanup;
    }

    //risk_function(data, &R, subgrad, NULL);
    risk_function(data, &R, subgrad, W);

    bmrm.nCP = 0;
    bmrm.nIter = 0;
    bmrm.exitflag = 0;

    b[0] = -R;
    LIBBMRM_MEMCPY(subgrad, A, nDim*sizeof(float64_t));

    /* Compute initial value of Fp, Fd, assuming that W is zero vector */
    sq_norm_W = 0;
    bmrm.Fp = R + 0.5*lambda*sq_norm_W;
    bmrm.Fd = -LIBBMRM_PLUS_INF;

    /* main loop */
    while (bmrm.exitflag == 0)
    {
        bmrm.nIter++;

        /* Update H */
        if (bmrm.nCP > 0)
        {
            for (uint32_t i = 0; i < bmrm.nCP - 1; i++)
            {
                rsum = 0.0;
                for (uint32_t j = 0; j < nDim; j++)
                {
                    rsum += A[LIBBMRM_INDEX(j, i, nDim)]*A[LIBBMRM_INDEX(j, bmrm.nCP, nDim)]/lambda;
                }
                H[LIBBMRM_INDEX(i, bmrm.nCP, BufSize)] = rsum;
            }
            for (uint32_t i = 0; i < bmrm.nCP - 1; i++)
            {
<<<<<<< HEAD
                H[LIBBMRM_INDEX(bmrm.nCP, i, BufSize)] = H[LIBBMRM_INDEX(i, bmrm.nCP, BufSize)];
=======
                H[LIBBMRM_INDEX(bmrm.nCP, i, BufSize)] = H[LIBBMRM_INDEX(i, bmrm.nCP)];
>>>>>>> 8f28ad3... BMRM work in progress
            }
        }
        H[LIBBMRM_INDEX(bmrm.nCP, bmrm.nCP, BufSize)] = 0.0;
        for (uint32_t i = 0; i < nDim; i++)
            H[LIBBMRM_INDEX(bmrm.nCP, bmrm.nCP, BufSize)] += A[LIBBMRM_INDEX(i, bmrm.nCP, nDim)]*A[LIBBMRM_INDEX(i, bmrm.nCP, nDim)]/lambda;


        diag_H[bmrm.nCP] = H[LIBBMRM_INDEX(bmrm.nCP, bmrm.nCP, BufSize)];

        bmrm.nCP++;

        /* call QP solver */
<<<<<<< HEAD
        qp_exitflag = libqp_splx_solver(&get_col, diag_H, b, &C, I, &S, beta,
=======
        qp_exitflag = libqp_splx_solver(&get_col, diag_H, b, 1.0, I, &S, beta,
>>>>>>> 8f28ad3... BMRM work in progress
                                        bmrm.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);

        bmrm.qp_exitflag = qp_exitflag.exitflag;

        /* W update */
        for (uint32_t i = 0; i < nDim; i++)
        {
            rsum = 0.0;
            for (uint32_t j = 0; j < bmrm.nCP; j++)
            {
                rsum += A[LIBBMRM_INDEX(i, j, nDim)]*beta[j]/lambda;
            }
            W[i] = -rsum;
        }

        /* risk and subgradient computation */
        risk_function(data, &R, subgrad, W);
        LIBBMRM_MEMCPY(subgrad, A+bmrm.nCP*nDim, nDim*sizeof(float64_t));
<<<<<<< HEAD
        b[bmrm.nCP] = -R;
        for (uint32_t j = 0; j < nDim; j++)
            b[bmrm.nCP] += subgrad[j]*W[j];

        sq_norm_W = 0;
        for (uint32_t j = 0; j < nDim; j++)
            sq_norm_W += W[j]*W[j];
=======
        b[bmrm.nCP] = R;
        for (uint32_t j = 0; j < nDim; j++)
            b[bmrm.nCP] -= subgrad[j]*W[j];

        norm_W = 0;
        for (uint32_t j = 0; j < nDim; j++)
            norm_W += W[j]*W[j];
>>>>>>> 8f28ad3... BMRM work in progress

        bmrm.Fp = R + 0.5*lambda*sq_norm_W;
        bmrm.Fd = -qp_exitflag.QP;

        /* Stopping conditions */
        if (bmrm.Fp - bmrm.Fd <= TolRel*LIBBMRM_ABS(bmrm.Fp)) bmrm.exitflag = 1;
        if (bmrm.Fp - bmrm.Fd <= TolAbs) bmrm.exitflag = 2;
        if (bmrm.nCP >= BufSize) bmrm.exitflag = -1;

    } /* end of main loop */

cleanup:

    LIBBMRM_FREE(H);
    LIBBMRM_FREE(b);
    LIBBMRM_FREE(beta);
    LIBBMRM_FREE(A);
    LIBBMRM_FREE(diag_H);
    LIBBMRM_FREE(I);

    return(bmrm);
}
}
