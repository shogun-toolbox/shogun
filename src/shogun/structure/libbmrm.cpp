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

#include <shogun/structure/libbmrm.h>
#include <shogun/lib/external/libqp.h>
#include <shogun/lib/Time.h>

namespace shogun
{
static const uint32_t QPSolverMaxIter=10000000;
static const float64_t epsilon=0.0000000001;

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
		bmrm_data_T*    data,
		float64_t*      W,
		float64_t       TolRel,
		float64_t       TolAbs,
		float64_t       lambda,
		uint32_t        _BufSize,
		bool            cleanICP,
		uint32_t        cleanAfter,
		float64_t       K,
		uint32_t        Tmax,
		CRiskFunction*  risk_function)
{
	bmrm_return_value_T bmrm = {0, 0, 0, 0, 0, 0, 0};
	libqp_state_T qp_exitflag;
	float64_t *b, *beta, *diag_H, sq_norm_W;
	float64_t R, *subgrad, *A, QPSolverTolRel, rsum, C=1.0;
	uint32_t *I, *ICPcounter, *ICPs, cntICP=0;
	uint8_t S = 1;
	uint32_t nDim=data->w_dim;
	CTime ttime;
	float64_t tstart, tstop;

	float64_t *b2, *beta2, *diag_H2, *A2, *H2;
	uint32_t *I2, *ICPcounter2, nCP_new=0, idx=0, idx2=0, icp_iter=0, icp_iter2=0;
	int32_t idx_icp=0, idx_icp2=0;
	bool flag1=true, flag2=true;

	tstart=ttime.cur_time_diff(false);

	BufSize=_BufSize;
	QPSolverTolRel=TolRel*0.5;

	H=NULL;
	b=NULL;
	beta=NULL;
	A=NULL;
	subgrad=NULL;
	diag_H=NULL;
	I=NULL;
	ICPcounter=NULL;
	ICPs=NULL;

	b2=NULL;
	beta2=NULL;
	H2=NULL;
	I2=NULL;
	ICPcounter2=NULL;
	A2=NULL;
	diag_H2=NULL;

	H=(float64_t*)LIBBMRM_CALLOC(BufSize*BufSize, sizeof(float64_t));
	if (H==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	A=(float64_t*)LIBBMRM_CALLOC(nDim*BufSize, sizeof(float64_t));
	if (A==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	b=(float64_t*)LIBBMRM_CALLOC(BufSize, sizeof(float64_t));
	if (b==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	beta=(float64_t*)LIBBMRM_CALLOC(BufSize, sizeof(float64_t));
	if (beta==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	subgrad=(float64_t*)LIBBMRM_CALLOC(nDim, sizeof(float64_t));
	if (subgrad==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	diag_H=(float64_t*)LIBBMRM_CALLOC(BufSize, sizeof(float64_t));
	if (diag_H==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	I=(uint32_t*)LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));
	if (I==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	ICPcounter=(uint32_t*)LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));
	if (ICPcounter==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	ICPs=(uint32_t*)LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));
	if (ICPs==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	/* Temporary buffers for ICP removal */
	b2=(float64_t*)LIBBMRM_CALLOC(BufSize, sizeof(float64_t));
	beta2=(float64_t*)LIBBMRM_CALLOC(BufSize, sizeof(float64_t));
	ICPcounter2=(uint32_t*)LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));
	I2=(uint32_t*)LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));
	diag_H2=(float64_t*)LIBBMRM_CALLOC(BufSize, sizeof(float64_t));
	A2=(float64_t*)LIBBMRM_CALLOC(nDim*BufSize, sizeof(float64_t));
	H2=(float64_t*)LIBBMRM_CALLOC(BufSize*BufSize, sizeof(float64_t));
	if (b2==NULL || beta2==NULL || ICPcounter2==NULL || I2==NULL || diag_H2==NULL || A2==NULL || H2==NULL)
	{
		bmrm.exitflag=-2;
	}

	/* Iinitial solution */
	risk_function->risk(data, &R, subgrad, W);

	bmrm.nCP=0;
	bmrm.nIter=0;
	bmrm.exitflag=0;

	b[0]=-R;
	LIBBMRM_MEMCPY(A, subgrad, nDim*sizeof(float64_t));

	/* Compute initial value of Fp, Fd, assuming that W is zero vector */
	sq_norm_W=0;
	bmrm.Fp=R+0.5*lambda*sq_norm_W;
	bmrm.Fd=-LIBBMRM_PLUS_INF;

	tstop=ttime.cur_time_diff(false);

	/* Verbose output */
	SG_SPRINT("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, R=%lf\n",
			bmrm.nIter, tstop-tstart, bmrm.Fp, bmrm.Fd, R);

	/* main loop */
	while (bmrm.exitflag==0)
	{
		tstart=ttime.cur_time_diff(false);
		bmrm.nIter++;

		/* Update H */
		if (bmrm.nCP>0)
		{
			for (uint32_t i=0; i<bmrm.nCP; ++i)
			{
				rsum=0.0;
				for (uint32_t j=0; j<nDim; ++j)
				{
					rsum+=A[LIBBMRM_INDEX(j, i, nDim)]*A[LIBBMRM_INDEX(j, bmrm.nCP, nDim)];
				}
				H[LIBBMRM_INDEX(i, bmrm.nCP, BufSize)]=rsum/lambda;
			}
			for (uint32_t i=0; i<bmrm.nCP; ++i)
			{
				H[LIBBMRM_INDEX(bmrm.nCP, i, BufSize)]=H[LIBBMRM_INDEX(i, bmrm.nCP, BufSize)];
			}
		}
		H[LIBBMRM_INDEX(bmrm.nCP, bmrm.nCP, BufSize)]=0.0;

		for (uint32_t i=0; i<nDim; ++i)
			H[LIBBMRM_INDEX(bmrm.nCP, bmrm.nCP, BufSize)]+=A[LIBBMRM_INDEX(i, bmrm.nCP, nDim)]*A[LIBBMRM_INDEX(i, bmrm.nCP, nDim)]/lambda;

		diag_H[bmrm.nCP]=H[LIBBMRM_INDEX(bmrm.nCP, bmrm.nCP, BufSize)];
		I[bmrm.nCP]=1;

		bmrm.nCP++;

		/* call QP solver */
		qp_exitflag = libqp_splx_solver(&get_col, diag_H, b, &C, I, &S, beta,
				bmrm.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);

		bmrm.qp_exitflag=qp_exitflag.exitflag;

		/* Update ICPcounter (add one to unused and reset used) + compute number of active CPs*/
		bmrm.nzA=0;
		for (uint32_t aaa=0; aaa<bmrm.nCP; ++aaa)
		{
			if (beta[aaa]>epsilon)
			{
				bmrm.nzA+=1;
				ICPcounter[aaa]=0;
			} else {
				ICPcounter[aaa]+=1;
			}
		}

		/* W update */
		for (uint32_t i=0; i<nDim; ++i)
		{
			rsum = 0.0;
			for (uint32_t j=0; j<bmrm.nCP; ++j)
			{
				rsum += A[LIBBMRM_INDEX(i, j, nDim)]*beta[j];
			}
			W[i] = -rsum/lambda;
		}

		/* risk and subgradient computation */
		risk_function->risk(data, &R, subgrad, W);
		LIBBMRM_MEMCPY(A+bmrm.nCP*nDim, subgrad, nDim*sizeof(float64_t));
		b[bmrm.nCP]=-R;
		for (uint32_t j=0; j<nDim; ++j)
			b[bmrm.nCP]+=subgrad[j]*W[j];

		sq_norm_W = 0;
		for (uint32_t j=0; j<nDim; ++j)
			sq_norm_W+=W[j]*W[j];

		bmrm.Fp=R+0.5*lambda*sq_norm_W;
		bmrm.Fd=-qp_exitflag.QP;

		/* Stopping conditions */
		if (bmrm.Fp - bmrm.Fd <= TolRel*LIBBMRM_ABS(bmrm.Fp)) bmrm.exitflag=1;
		if (bmrm.Fp - bmrm.Fd <= TolAbs) bmrm.exitflag=2;
		if (bmrm.nCP >= BufSize) bmrm.exitflag=-1;

		tstop=ttime.cur_time_diff(false);

		/* Verbose output */
		SG_SPRINT("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, (Fp-Fd)=%lf, (Fp-Fd)/Fp=%lf, R=%lf, nCP=%d, nzA=%d\n",
				bmrm.nIter, tstop-tstart, bmrm.Fp, bmrm.Fd, bmrm.Fp-bmrm.Fd, (bmrm.Fp-bmrm.Fd)/bmrm.Fp, R, bmrm.nCP, bmrm.nzA);

		/* Check size of Buffer */
		if (bmrm.nCP>=BufSize)
		{
			bmrm.exitflag=-2;
			SG_SERROR("Buffer exceeded.\n");
		}

		/* Inactive Cutting Planes (ICP) removal */
		if (cleanICP)
		{
			/* find ICP */
			cntICP = 0;
			for (uint32_t aaa=0; aaa<bmrm.nCP; ++aaa)
				if (ICPcounter[aaa]>=cleanAfter)
				{
					ICPs[cntICP++]=aaa;
				}

			/* do ICP */
			if (cntICP > 0)
			{
				nCP_new=bmrm.nCP-cntICP;

				idx=0;
				idx2=0;
				icp_iter=0;
				icp_iter2=0;
				idx_icp=ICPs[icp_iter];
				idx_icp2=ICPs[icp_iter2];
				flag1=true; flag2=true;

				for (uint32_t i=0; i<bmrm.nCP; ++i)
				{
					if ((int32_t)i != idx_icp)
					{
						b2[idx]=b[i];
						beta2[idx]=beta[i];
						ICPcounter2[idx]=ICPcounter[i];
						I2[idx]=I[i];
						diag_H2[idx]=diag_H[i];
						LIBBMRM_MEMCPY(A2+idx*nDim, A+i*nDim, nDim*sizeof(float64_t));

						idx2=0;
						icp_iter2=0;
						idx_icp2=ICPs[icp_iter2];
						flag2=true;
						for (uint32_t j=0; j<bmrm.nCP; ++j)
						{
							if ((uint32_t)j != idx_icp2)
							{
								H2[LIBBMRM_INDEX(idx, idx2, BufSize)]=H[LIBBMRM_INDEX(i, j, BufSize)];
								idx2++;
							} else {
								if (flag2 && icp_iter2+1 < cntICP)
								{
									idx_icp2=ICPs[++icp_iter2];
								} else {
									flag2=false;
									idx_icp2=-1;
								}
							}
						}

						idx++;
					} else {
						if (flag1 && icp_iter+1 < cntICP)
						{
							idx_icp=ICPs[++icp_iter];
						} else {
							flag1=false;
							idx_icp=-1;
						}
					}
				}

				/* copy data from tmps back to original */
				LIBBMRM_MEMCPY(b, b2, nCP_new*sizeof(float64_t));
				LIBBMRM_MEMCPY(beta, beta2, nCP_new*sizeof(float64_t));
				LIBBMRM_MEMCPY(ICPcounter, ICPcounter2, nCP_new*sizeof(uint32_t));
				LIBBMRM_MEMCPY(I, I2, nCP_new*sizeof(uint32_t));
				LIBBMRM_MEMCPY(diag_H, diag_H2, nCP_new*sizeof(float64_t));
				LIBBMRM_MEMCPY(A, A2, nDim*nCP_new*sizeof(float64_t));
				for (uint32_t i=0; i<nCP_new; ++i)
					for (uint32_t j=0; j<nCP_new; ++j)
						H[LIBBMRM_INDEX(i, j, BufSize)]=H2[LIBBMRM_INDEX(i, j, BufSize)];

				bmrm.nCP=nCP_new;
			}
		}

	} /* end of main loop */

cleanup:

	LIBBMRM_FREE(H);
	LIBBMRM_FREE(b);
	LIBBMRM_FREE(beta);
	LIBBMRM_FREE(A);
	LIBBMRM_FREE(subgrad);
	LIBBMRM_FREE(diag_H);
	LIBBMRM_FREE(I);
	LIBBMRM_FREE(ICPcounter);
	LIBBMRM_FREE(ICPs);

	LIBBMRM_FREE(H2);
	LIBBMRM_FREE(b2);
	LIBBMRM_FREE(beta2);
	LIBBMRM_FREE(A2);
	LIBBMRM_FREE(diag_H2);
	LIBBMRM_FREE(I2);
	LIBBMRM_FREE(ICPcounter2);

	return(bmrm);
}
}
