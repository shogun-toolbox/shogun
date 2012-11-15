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
static const uint32_t QPSolverMaxIter=0xFFFFFFFF;
static const float64_t epsilon=0.0;

static float64_t *H;
static uint32_t BufSize;

void add_cutting_plane(
		bmrm_ll**	tail,
		bool*		map,
		float64_t*	A,
		uint32_t 	free_idx,
		float64_t*	cp_data,
		uint32_t 	dim)
{
	ASSERT(map[free_idx]);

	LIBBMRM_MEMCPY(A+free_idx*dim, cp_data, dim*sizeof(float64_t));
	map[free_idx]=false;

	bmrm_ll *cp=(bmrm_ll*)LIBBMRM_CALLOC(1, sizeof(bmrm_ll));

	if (cp==NULL)
	{
		SG_SERROR("Out of memory.\n");
		return;
	}

	cp->address=A+(free_idx*dim);
	cp->prev=*tail;
	cp->next=NULL;
	cp->idx=free_idx;
	(*tail)->next=cp;
	*tail=cp;
}

void remove_cutting_plane(
		bmrm_ll**	head,
		bmrm_ll**	tail,
		bool* 		map,
		float64_t*	icp)
{
	bmrm_ll *cp_list_ptr=*head;

	while(cp_list_ptr->address != icp)
	{
		cp_list_ptr=cp_list_ptr->next;
	}

	if (cp_list_ptr==*head)
	{
		*head=(*head)->next;
		cp_list_ptr->next->prev=NULL;
	}
	else if (cp_list_ptr==*tail)
	{
		*tail=(*tail)->prev;
		cp_list_ptr->prev->next=NULL;
	}
	else
	{
		cp_list_ptr->prev->next=cp_list_ptr->next;
		cp_list_ptr->next->prev=cp_list_ptr->prev;
	}

	map[cp_list_ptr->idx]=true;
	LIBBMRM_FREE(cp_list_ptr);
}

/*----------------------------------------------------------------------
  Returns pointer at i-th column of Hessian matrix.
  ----------------------------------------------------------------------*/
static const float64_t *get_col( uint32_t i)
{
	return( &H[ BufSize*i ] );
}

bmrm_return_value_T svm_bmrm_solver(
		CStructuredModel* model,
		float64_t*       W,
		float64_t        TolRel,
		float64_t        TolAbs,
		float64_t        _lambda,
		uint32_t         _BufSize,
		bool             cleanICP,
		uint32_t         cleanAfter,
		float64_t        K,
		uint32_t         Tmax,
		bool             verbose)
{
	bmrm_return_value_T bmrm;
	libqp_state_T qp_exitflag={0, 0, 0, 0};
	float64_t *b, *beta, *diag_H, *prevW;
	float64_t R, *subgrad, *A, QPSolverTolRel, C=1.0, wdist=0.0;
	floatmax_t rsum, sq_norm_W, sq_norm_Wdiff=0.0;
	uint32_t *I, *ICPcounter, *ACPs, cntICP=0, cntACP=0;
	uint8_t S=1;
	uint32_t nDim=model->get_dim();
	float64_t **ICPs;

	CTime ttime;
	float64_t tstart, tstop;

	uint32_t nCP_new=0;

	bmrm_ll *CPList_head, *CPList_tail, *cp_ptr, *cp_ptr2, *cp_list=NULL;
	float64_t *A_1=NULL, *A_2=NULL, *H_buff;
	bool *map=NULL;


	tstart=ttime.cur_time_diff(false);

	BufSize=_BufSize;
	QPSolverTolRel=1e-9;

	H=NULL;
	b=NULL;
	beta=NULL;
	A=NULL;
	subgrad=NULL;
	diag_H=NULL;
	I=NULL;
	ICPcounter=NULL;
	ICPs=NULL;
	ACPs=NULL;
	H_buff=NULL;
	prevW=NULL;

	H= (float64_t*) LIBBMRM_CALLOC(BufSize*BufSize, sizeof(float64_t));

	if (H==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	A= (float64_t*) LIBBMRM_CALLOC(nDim*BufSize, sizeof(float64_t));

	if (A==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	b= (float64_t*) LIBBMRM_CALLOC(BufSize, sizeof(float64_t));

	if (b==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	beta= (float64_t*) LIBBMRM_CALLOC(BufSize, sizeof(float64_t));

	if (beta==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	subgrad= (float64_t*) LIBBMRM_CALLOC(nDim, sizeof(float64_t));

	if (subgrad==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	diag_H= (float64_t*) LIBBMRM_CALLOC(BufSize, sizeof(float64_t));

	if (diag_H==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	I= (uint32_t*) LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));

	if (I==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	ICPcounter= (uint32_t*) LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));

	if (ICPcounter==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	ICPs= (float64_t**) LIBBMRM_CALLOC(BufSize, sizeof(float64_t*));

	if (ICPs==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	ACPs= (uint32_t*) LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));

	if (ACPs==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	map= (bool*) LIBBMRM_CALLOC(BufSize, sizeof(bool));

	if (map==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	memset( (bool*) map, true, BufSize);

	cp_list= (bmrm_ll*) LIBBMRM_CALLOC(1, sizeof(bmrm_ll));

	if (cp_list==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	/* Temporary buffers for ICP removal */
	H_buff= (float64_t*) LIBBMRM_CALLOC(BufSize*BufSize, sizeof(float64_t));

	if (H_buff==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	prevW= (float64_t*) LIBBMRM_CALLOC(nDim, sizeof(float64_t));

	if (prevW==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	bmrm.hist_Fp = SGVector< float64_t >(BufSize);
	bmrm.hist_Fd = SGVector< float64_t >(BufSize);
	bmrm.hist_wdist = SGVector< float64_t >(BufSize);

	/* Iinitial solution */
	R=model->risk(subgrad, W);

	bmrm.nCP=0;
	bmrm.nIter=0;
	bmrm.exitflag=0;

	b[0]=-R;

	/* Cutting plane auxiliary double linked list */

	LIBBMRM_MEMCPY(A, subgrad, nDim*sizeof(float64_t));
	map[0]=false;
	cp_list->address=&A[0];
	cp_list->idx=0;
	cp_list->prev=NULL;
	cp_list->next=NULL;
	CPList_head=cp_list;
	CPList_tail=cp_list;

	/* Compute initial value of Fp, Fd, assuming that W is zero vector */

	sq_norm_W=0;
	bmrm.Fp=R+0.5*_lambda*sq_norm_W;
	bmrm.Fd=-LIBBMRM_PLUS_INF;

	tstop=ttime.cur_time_diff(false);

	/* Verbose output */

	if (verbose)
		SG_SPRINT("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, R=%lf\n",
				bmrm.nIter, tstop-tstart, bmrm.Fp, bmrm.Fd, R);

	/* store Fp, Fd and wdist history */
	bmrm.hist_Fp[0]=bmrm.Fp;
	bmrm.hist_Fd[0]=bmrm.Fd;
	bmrm.hist_wdist[0]=0.0;

	/* main loop */

	while (bmrm.exitflag==0)
	{
		tstart=ttime.cur_time_diff(false);
		bmrm.nIter++;

		/* Update H */

		if (bmrm.nCP>0)
		{
			A_2=get_cutting_plane(CPList_tail);
			cp_ptr=CPList_head;

			for (uint32_t i=0; i<bmrm.nCP; ++i)
			{
				A_1=get_cutting_plane(cp_ptr);
				cp_ptr=cp_ptr->next;
				rsum=0.0;

				for (uint32_t j=0; j<nDim; ++j)
				{
					rsum+=A_1[j]*A_2[j];
				}

				H[LIBBMRM_INDEX(i, bmrm.nCP, BufSize)]=rsum/_lambda;
			}

			for (uint32_t i=0; i<bmrm.nCP; ++i)
			{
				H[LIBBMRM_INDEX(bmrm.nCP, i, BufSize)]=
					H[LIBBMRM_INDEX(i, bmrm.nCP, BufSize)];
			}
		}

		rsum=0.0;
		A_2=get_cutting_plane(CPList_tail);

		for (uint32_t i=0; i<nDim; ++i)
			rsum+=A_2[i]*A_2[i];

		H[LIBBMRM_INDEX(bmrm.nCP, bmrm.nCP, BufSize)]=rsum/_lambda;

		diag_H[bmrm.nCP]=H[LIBBMRM_INDEX(bmrm.nCP, bmrm.nCP, BufSize)];
		I[bmrm.nCP]=1;

		bmrm.nCP++;
		beta[bmrm.nCP]=0.0; // [beta; 0]

		/* call QP solver */
		qp_exitflag=libqp_splx_solver(&get_col, diag_H, b, &C, I, &S, beta,
				bmrm.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);

		bmrm.qp_exitflag=qp_exitflag.exitflag;

		/* Update ICPcounter (add one to unused and reset used)
		 * + compute number of active CPs */
		bmrm.nzA=0;

		for (uint32_t aaa=0; aaa<bmrm.nCP; ++aaa)
		{
			if (beta[aaa]>epsilon)
			{
				++bmrm.nzA;
				ICPcounter[aaa]=0;
			}
			else
			{
				ICPcounter[aaa]+=1;
			}
		}

		/* W update */
		for (uint32_t i=0; i<nDim; ++i)
		{
			rsum=0.0;
			cp_ptr=CPList_head;

			for (uint32_t j=0; j<bmrm.nCP; ++j)
			{
				A_1=get_cutting_plane(cp_ptr);
				cp_ptr=cp_ptr->next;
				rsum+=A_1[i]*beta[j];
			}

			W[i]=-rsum/_lambda;
		}

		/* risk and subgradient computation */
		R = model->risk(subgrad, W);
		b[bmrm.nCP]=-R;
		add_cutting_plane(&CPList_tail, map, A,
				find_free_idx(map, BufSize), subgrad, nDim);

		sq_norm_W=0.0;
		sq_norm_Wdiff=0.0;

		for (uint32_t j=0; j<nDim; ++j)
		{
			b[bmrm.nCP]+=subgrad[j]*W[j];
			sq_norm_W+=W[j]*W[j];
			sq_norm_Wdiff+=(W[j]-prevW[j])*(W[j]-prevW[j]);
		}

		bmrm.Fp=R+0.5*_lambda*sq_norm_W;
		bmrm.Fd=-qp_exitflag.QP;
		wdist=CMath::sqrt(sq_norm_Wdiff);

		/* Stopping conditions */

		if (bmrm.Fp - bmrm.Fd <= TolRel*LIBBMRM_ABS(bmrm.Fp))
			bmrm.exitflag=1;

		if (bmrm.Fp - bmrm.Fd <= TolAbs)
			bmrm.exitflag=2;

		if (bmrm.nCP >= BufSize)
			bmrm.exitflag=-1;

		tstop=ttime.cur_time_diff(false);

		/* Verbose output */

		if (verbose)
			SG_SPRINT("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, (Fp-Fd)=%lf, (Fp-Fd)/Fp=%lf, R=%lf, nCP=%d, nzA=%d, QPexitflag=%d\n",
					bmrm.nIter, tstop-tstart, bmrm.Fp, bmrm.Fd, bmrm.Fp-bmrm.Fd,
					(bmrm.Fp-bmrm.Fd)/bmrm.Fp, R, bmrm.nCP, bmrm.nzA, qp_exitflag.exitflag);

		/* Keep Fp, Fd and w_dist history */
		bmrm.hist_Fp[bmrm.nIter]=bmrm.Fp;
		bmrm.hist_Fd[bmrm.nIter]=bmrm.Fd;
		bmrm.hist_wdist[bmrm.nIter]=wdist;

		/* Check size of Buffer */

		if (bmrm.nCP>=BufSize)
		{
			bmrm.exitflag=-2;
			SG_SERROR("Buffer exceeded.\n");
		}

		/* keep W (for wdist history track) */
		LIBBMRM_MEMCPY(prevW, W, nDim*sizeof(float64_t));

		/* Inactive Cutting Planes (ICP) removal */
		if (cleanICP)
		{
			/* find ICP */
			cntICP=0;
			cntACP=0;
			cp_ptr=CPList_head;
			uint32_t tmp_idx=0;

			while (cp_ptr != CPList_tail)
			{
				if (ICPcounter[tmp_idx++]>=cleanAfter)
				{
					ICPs[cntICP++]=cp_ptr->address;
				}
				else
				{
					ACPs[cntACP++]=tmp_idx-1;
				}

				cp_ptr=cp_ptr->next;
			}

			/* do ICP removal */
			if (cntICP > 0)
			{
				nCP_new=bmrm.nCP-cntICP;

				for (uint32_t i=0; i<cntICP; ++i)
				{
					tmp_idx=0;
					cp_ptr=CPList_head;

					while(cp_ptr->address != ICPs[i])
					{
						cp_ptr=cp_ptr->next;
						tmp_idx++;
					}

					remove_cutting_plane(&CPList_head, &CPList_tail, map, ICPs[i]);

					LIBBMRM_MEMMOVE(b+tmp_idx, b+tmp_idx+1,
							(bmrm.nCP-tmp_idx)*sizeof(float64_t));
					LIBBMRM_MEMMOVE(beta+tmp_idx, beta+tmp_idx+1,
							(bmrm.nCP-tmp_idx)*sizeof(float64_t));
					LIBBMRM_MEMMOVE(diag_H+tmp_idx, diag_H+tmp_idx+1,
							(bmrm.nCP-tmp_idx)*sizeof(float64_t));
					LIBBMRM_MEMMOVE(I+tmp_idx, I+tmp_idx+1,
							(bmrm.nCP-tmp_idx)*sizeof(uint32_t));
					LIBBMRM_MEMMOVE(ICPcounter+tmp_idx, ICPcounter+tmp_idx+1,
							(bmrm.nCP-tmp_idx)*sizeof(uint32_t));
				}

				/* H */
				for (uint32_t i=0; i < nCP_new; ++i)
				{
					for (uint32_t j=0; j < nCP_new; ++j)
					{
						H_buff[LIBBMRM_INDEX(i, j, BufSize)]=
							H[LIBBMRM_INDEX(ACPs[i], ACPs[j], BufSize)];
					}
				}

				for (uint32_t i=0; i<nCP_new; ++i)
					for (uint32_t j=0; j<nCP_new; ++j)
						H[LIBBMRM_INDEX(i, j, BufSize)]=
							H_buff[LIBBMRM_INDEX(i, j, BufSize)];

				bmrm.nCP=nCP_new;
			}
		}
	} /* end of main loop */

	bmrm.hist_Fp.resize_vector(bmrm.nIter);
	bmrm.hist_Fd.resize_vector(bmrm.nIter);
	bmrm.hist_wdist.resize_vector(bmrm.nIter);

	cp_ptr=CPList_head;

	while(cp_ptr!=NULL)
	{
		cp_ptr2=cp_ptr;
		cp_ptr=cp_ptr->next;
		LIBBMRM_FREE(cp_ptr2);
		cp_ptr2=NULL;
	}

	cp_list=NULL;

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
	LIBBMRM_FREE(ACPs);
	LIBBMRM_FREE(H_buff);
	LIBBMRM_FREE(map);
	LIBBMRM_FREE(prevW);

	if (cp_list)
		LIBBMRM_FREE(cp_list);

	return(bmrm);
}
}
