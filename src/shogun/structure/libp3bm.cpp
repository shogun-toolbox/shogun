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
 * Implementation of the Proximal Point P-BMRM (p3bm)
 *--------------------------------------------------------------------- */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <shogun/structure/libppbm.h>
#include <shogun/lib/external/libqp.h>
#include <shogun/lib/Time.h>

namespace shogun
{
static const uint32_t QPSolverMaxIter=0xFFFFFFFF;
static const float64_t epsilon=0.0;

static float64_t *H, *H2;
static uint32_t BufSize;

/*----------------------------------------------------------------------
  Returns pointer at i-th column of Hessian matrix.
  ----------------------------------------------------------------------*/
static const float64_t *get_col( uint32_t i)
{
	return( &H2[ BufSize*i ] );
}

bmrm_return_value_T svm_p3bm_solver(
		CStructuredModel* model,
		float64_t*      W,
		float64_t       TolRel,
		float64_t       TolAbs,
		float64_t       lambda,
		uint32_t        _BufSize,
		bool            cleanICP,
		uint32_t        cleanAfter,
		float64_t       K,
		uint32_t        Tmax,
		uint32_t        cp_models,
		bool            verbose)
{
	bmrm_return_value_T p3bmrm={0, 0, 0, 0, 0, 0, 0};
	libqp_state_T qp_exitflag={0, 0, 0, 0}, qp_exitflag_good={0, 0, 0, 0};
	float64_t *b, *b2, *beta, *beta_good, *beta_start, *diag_H, *diag_H2;
	float64_t R, *Rt, **subgrad_t, *A, QPSolverTolRel, *C=NULL;
	float64_t *prevW, *wt, alpha, alpha_start, alpha_good=0.0, Fd_alpha0=0.0;
	float64_t lastFp, wdist, gamma=0.0;
	floatmax_t rsum, sq_norm_W, sq_norm_Wdiff, sq_norm_prevW, eps;
	uint32_t *I, *I2, *I_start, *I_good, *ICPcounter, *ACPs, cntICP=0, cntACP=0;
	uint8_t *S=NULL;
	uint32_t nCP_new=0, qp_cnt=0;
	bmrm_ll *CPList_head, *CPList_tail, *cp_ptr, *cp_ptr2, *cp_list=NULL;
	float64_t *A_1=NULL, *H_buff, **ICPs;
	bool *map=NULL, tuneAlpha=true, flag=true;
	bool alphaChanged=false, isThereGoodSolution=false;
	TMultipleCPinfo **info=NULL;
	uint32_t nDim=model->get_dim();
	uint32_t to=0, N=0, cp_i=0;

	CTime ttime;
	float64_t tstart, tstop;


	tstart=ttime.cur_time_diff(false);

	BufSize=_BufSize*cp_models;
	QPSolverTolRel=1e-9;

	H=NULL;
	b=NULL;
	beta=NULL;
	A=NULL;
	subgrad_t=NULL;
	diag_H=NULL;
	I=NULL;
	ICPcounter=NULL;
	ICPs=NULL;
	ACPs=NULL;
	prevW=NULL;
	wt=NULL;
	H_buff=NULL;
	diag_H2=NULL;
	b2=NULL;
	I2=NULL;
	H2=NULL;
	I_good=NULL;
	I_start=NULL;
	beta_start=NULL;
	beta_good=NULL;

	alpha=0.0;

	H= (float64_t*) LIBBMRM_CALLOC(BufSize*BufSize, sizeof(float64_t));

	A= (float64_t*) LIBBMRM_CALLOC(nDim*BufSize, sizeof(float64_t));

	b= (float64_t*) LIBBMRM_CALLOC(BufSize, sizeof(float64_t));

	beta= (float64_t*) LIBBMRM_CALLOC(BufSize, sizeof(float64_t));

	subgrad_t= (float64_t**) LIBBMRM_CALLOC(cp_models, sizeof(float64_t*));

	Rt= (float64_t*) LIBBMRM_CALLOC(cp_models, sizeof(float64_t));

	diag_H= (float64_t*) LIBBMRM_CALLOC(BufSize, sizeof(float64_t));

	I= (uint32_t*) LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));

	ICPcounter= (uint32_t*) LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));

	ICPs= (float64_t**) LIBBMRM_CALLOC(BufSize, sizeof(float64_t*));

	ACPs= (uint32_t*) LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));

	cp_list= (bmrm_ll*) LIBBMRM_CALLOC(1, sizeof(bmrm_ll));

	prevW= (float64_t*) LIBBMRM_CALLOC(nDim, sizeof(float64_t));

	wt= (float64_t*) LIBBMRM_CALLOC(nDim, sizeof(float64_t));

	C= (float64_t*) LIBBMRM_CALLOC(cp_models, sizeof(float64_t));

	S= (uint8_t*) LIBBMRM_CALLOC(cp_models, sizeof(uint8_t));

	info= (TMultipleCPinfo**) LIBBMRM_CALLOC(cp_models, sizeof(TMultipleCPinfo*));

	int32_t num_feats = model->get_features()->get_num_vectors();

	if (H==NULL || A==NULL || b==NULL || beta==NULL || subgrad_t==NULL ||
			diag_H==NULL || I==NULL || ICPcounter==NULL || ICPs==NULL || ACPs==NULL ||
			cp_list==NULL || prevW==NULL || wt==NULL || Rt==NULL || C==NULL ||
			S==NULL || info==NULL)
	{
		p3bmrm.exitflag=-2;
		goto cleanup;
	}

	/* multiple cutting plane model init */

	to=0;
	N= (uint32_t) round( (float64_t) ((float64_t)num_feats / (float64_t) cp_models));

	for (uint32_t p=0; p<cp_models; ++p)
	{
		S[p]=1;
		C[p]=1.0;
		info[p]=(TMultipleCPinfo*)LIBBMRM_CALLOC(1, sizeof(TMultipleCPinfo));
		subgrad_t[p]=(float64_t*)LIBBMRM_CALLOC(nDim, sizeof(float64_t));

		if (subgrad_t[p]==NULL || info[p]==NULL)
		{
			p3bmrm.exitflag=-2;
			goto cleanup;
		}

		info[p]->from=to;
		to=((p+1)*N > num_feats) ? num_feats : (p+1)*N;
		info[p]->N=to-info[p]->from;
	}

	map= (bool*) LIBBMRM_CALLOC(BufSize, sizeof(bool));

	if (map==NULL)
	{
		p3bmrm.exitflag=-2;
		goto cleanup;
	}

	memset( (bool*) map, true, BufSize);

	/* Temporary buffers for ICP removal */
	H_buff= (float64_t*) LIBBMRM_CALLOC(BufSize*BufSize, sizeof(float64_t));

	if (H_buff==NULL)
	{
		p3bmrm.exitflag=-2;
		goto cleanup;
	}

	/* Temporary buffers */
	beta_start= (float64_t*) LIBBMRM_CALLOC(BufSize, sizeof(float64_t));

	beta_good= (float64_t*) LIBBMRM_CALLOC(BufSize, sizeof(float64_t));

	b2= (float64_t*) LIBBMRM_CALLOC(BufSize, sizeof(float64_t));

	diag_H2= (float64_t*) LIBBMRM_CALLOC(BufSize, sizeof(float64_t));

	H2= (float64_t*) LIBBMRM_CALLOC(BufSize*BufSize, sizeof(float64_t));

	I_start= (uint32_t*) LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));

	I_good= (uint32_t*) LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));

	I2= (uint32_t*) LIBBMRM_CALLOC(BufSize, sizeof(uint32_t));

	if (beta_start==NULL || beta_good==NULL || b2==NULL || diag_H2==NULL ||
			I_start==NULL || I_good==NULL || I2==NULL || H2==NULL)
	{
		p3bmrm.exitflag=-2;
		goto cleanup;
	}

	/* Iinitial solution */
	Rt[0] = model->risk(subgrad_t[0], W, info[0]);

	p3bmrm.nCP=0;
	p3bmrm.nIter=0;
	p3bmrm.exitflag=0;

	b[0]=-Rt[0];

	/* Cutting plane auxiliary double linked list */
	LIBBMRM_MEMCPY(A, subgrad_t[0], nDim*sizeof(float64_t));
	map[0]=false;
	cp_list->address=&A[0];
	cp_list->idx=0;
	cp_list->prev=NULL;
	cp_list->next=NULL;
	CPList_head=cp_list;
	CPList_tail=cp_list;

	for (uint32_t p=1; p<cp_models; ++p)
	{
		Rt[p] = model->risk(subgrad_t[p], W, info[p]);
		b[p]=-Rt[p];
		add_cutting_plane(&CPList_tail, map, A, find_free_idx(map, BufSize), subgrad_t[p], nDim);
	}

	/* Compute initial value of Fp, Fd, assuming that W is zero vector */
	R=0.0;

	for (uint32_t p=0; p<cp_models; ++p)
		R+=Rt[p];

	sq_norm_W=0.0;
	sq_norm_Wdiff=0.0;

	for (uint32_t j=0; j<nDim; ++j)
	{
		for (uint32_t p=0; p<cp_models; ++p)
			b[p]+=subgrad_t[p][j]*W[j];

		sq_norm_W+=W[j]*W[j];
		sq_norm_Wdiff+=(W[j]-prevW[j])*(W[j]-prevW[j]);
	}

	p3bmrm.Fp=R+0.5*lambda*sq_norm_W + alpha*sq_norm_Wdiff;
	p3bmrm.Fd=-LIBBMRM_PLUS_INF;
	lastFp=p3bmrm.Fp;

	/* if there is initial W, then set K to be 0.01 times its norm */
	K = (sq_norm_W == 0.0) ? 0.4 : 0.01*::sqrt(sq_norm_W);

	LIBBMRM_MEMCPY(prevW, W, nDim*sizeof(float64_t));

	tstop=ttime.cur_time_diff(false);

	/* Verbose output */
	if (verbose)
		SG_SPRINT("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, R=%lf, K=%lf, CPmodels=%d\n",
				p3bmrm.nIter, tstop-tstart, p3bmrm.Fp, p3bmrm.Fd, R, K, cp_models);

	/* main loop */
	while (p3bmrm.exitflag==0)
	{
		tstart=ttime.cur_time_diff(false);
		p3bmrm.nIter++;

		/* Update H */
		if (p3bmrm.nIter==1)
		{
			cp_ptr=CPList_head;

			for (cp_i=0; cp_i<cp_models; ++cp_i)  /* for all cutting planes */
			{
				A_1=get_cutting_plane(cp_ptr);

				for (uint32_t p=0; p<cp_models; ++p)
				{
					rsum=0.0;

					for (uint32_t j=0; j<nDim; ++j)
					{
						rsum+=subgrad_t[p][j]*A_1[j];
					}

					H[LIBBMRM_INDEX(p, cp_i, BufSize)]=rsum;
				}

				cp_ptr=cp_ptr->next;
			}
		}
		else
		{
			cp_ptr=CPList_head;

			for (cp_i=0; cp_i<p3bmrm.nCP+cp_models; ++cp_i)  /* for all cutting planes */
			{
				A_1=get_cutting_plane(cp_ptr);

				for (uint32_t p=0; p<cp_models; ++p)
				{
					rsum=0.0;

					for (uint32_t j=0; j<nDim; ++j)
						rsum+=subgrad_t[p][j]*A_1[j];

					H[LIBBMRM_INDEX(p3bmrm.nCP+p, cp_i, BufSize)]=rsum;
				}

				cp_ptr=cp_ptr->next;
			}

			for (uint32_t i=0; i<p3bmrm.nCP; ++i)
				for (uint32_t j=0; j<cp_models; ++j)
					H[LIBBMRM_INDEX(i, p3bmrm.nCP+j, BufSize)]=
						H[LIBBMRM_INDEX(p3bmrm.nCP+j, i, BufSize)];
		}

		for (uint32_t p=0; p<cp_models; ++p)
			diag_H[p3bmrm.nCP+p]=H[LIBBMRM_INDEX(p3bmrm.nCP+p, p3bmrm.nCP+p, BufSize)];

		p3bmrm.nCP+=cp_models;

		/* tune alpha cycle */
		/* ------------------------------------------------------------------------ */
		flag=true;
		isThereGoodSolution=false;

		for (uint32_t p=0; p<cp_models; ++p)
		{
			I[p3bmrm.nCP-cp_models+p]=p+1;
			beta[p3bmrm.nCP-cp_models+p]=0.0;
		}

		LIBBMRM_MEMCPY(beta_start, beta, p3bmrm.nCP*sizeof(float64_t));
		LIBBMRM_MEMCPY(I_start, I, p3bmrm.nCP*sizeof(uint32_t));
		qp_cnt=0;

		if (tuneAlpha)
		{
			alpha_start=alpha; alpha=0.0;
			LIBBMRM_MEMCPY(I2, I_start, p3bmrm.nCP*sizeof(uint32_t));

			/* add alpha-dependent terms to H, diag_h and b */
			cp_ptr=CPList_head;

			for (uint32_t i=0; i<p3bmrm.nCP; ++i)
			{
				rsum=0.0;
				A_1=get_cutting_plane(cp_ptr);
				cp_ptr=cp_ptr->next;

				for (uint32_t j=0; j<nDim; ++j)
					rsum+=A_1[j]*prevW[j];

				b2[i]=b[i]-((2*alpha)/(lambda+2*alpha))*rsum;
				diag_H2[i]=diag_H[i]/(lambda+2*alpha);

				for (uint32_t j=0; j<p3bmrm.nCP; ++j)
					H2[LIBBMRM_INDEX(i, j, BufSize)]=
						H[LIBBMRM_INDEX(i, j, BufSize)]/(lambda+2*alpha);

			}

			/* solve QP with current alpha */
			qp_exitflag=libqp_splx_solver(&get_col, diag_H2, b2, C, I2, S, beta,
					p3bmrm.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
			p3bmrm.qp_exitflag=qp_exitflag.exitflag;
			qp_cnt++;
			Fd_alpha0=-qp_exitflag.QP;

			/* obtain w_t and check if norm(w_{t+1} -w_t) <= K */
			for (uint32_t i=0; i<nDim; ++i)
			{
				rsum=0.0;
				cp_ptr=CPList_head;

				for (uint32_t j=0; j<p3bmrm.nCP; ++j)
				{
					A_1=get_cutting_plane(cp_ptr);
					cp_ptr=cp_ptr->next;
					rsum+=A_1[i]*beta[j];
				}

				wt[i]=(2*alpha*prevW[i] - rsum)/(lambda+2*alpha);
			}

			sq_norm_Wdiff=0.0;

			for (uint32_t i=0; i<nDim; ++i)
				sq_norm_Wdiff+=(wt[i]-prevW[i])*(wt[i]-prevW[i]);

			if (::sqrt(sq_norm_Wdiff) <= K)
			{
				flag=false;

				if (alpha!=alpha_start)
					alphaChanged=true;
			}
			else
			{
				alpha=alpha_start;
			}

			while(flag)
			{
				LIBBMRM_MEMCPY(I2, I_start, p3bmrm.nCP*sizeof(uint32_t));
				LIBBMRM_MEMCPY(beta, beta_start, p3bmrm.nCP*sizeof(float64_t));

				/* add alpha-dependent terms to H, diag_h and b */
				cp_ptr=CPList_head;

				for (uint32_t i=0; i<p3bmrm.nCP; ++i)
				{
					rsum=0.0;
					A_1=get_cutting_plane(cp_ptr);
					cp_ptr=cp_ptr->next;

					for (uint32_t j=0; j<nDim; ++j)
						rsum+=A_1[j]*prevW[j];

					b2[i]=b[i]-((2*alpha)/(lambda+2*alpha))*rsum;
					diag_H2[i]=diag_H[i]/(lambda+2*alpha);

					for (uint32_t j=0; j<p3bmrm.nCP; ++j)
						H2[LIBBMRM_INDEX(i, j, BufSize)]=H[LIBBMRM_INDEX(i, j, BufSize)]/(lambda+2*alpha);
				}

				/* solve QP with current alpha */
				qp_exitflag=libqp_splx_solver(&get_col, diag_H2, b2, C, I2, S, beta,
						p3bmrm.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
				p3bmrm.qp_exitflag=qp_exitflag.exitflag;
				qp_cnt++;

				/* obtain w_t and check if norm(w_{t+1}-w_t) <= K */
				for (uint32_t i=0; i<nDim; ++i)
				{
					rsum=0.0;
					cp_ptr=CPList_head;

					for (uint32_t j=0; j<p3bmrm.nCP; ++j)
					{
						A_1=get_cutting_plane(cp_ptr);
						cp_ptr=cp_ptr->next;
						rsum+=A_1[i]*beta[j];
					}

					wt[i]=(2*alpha*prevW[i] - rsum)/(lambda+2*alpha);
				}

				sq_norm_Wdiff=0.0;

				for (uint32_t i=0; i<nDim; ++i)
					sq_norm_Wdiff+=(wt[i]-prevW[i])*(wt[i]-prevW[i]);

				if (::sqrt(sq_norm_Wdiff) > K)
				{
					/* if there is a record of some good solution (i.e. adjust alpha by division by 2) */

					if (isThereGoodSolution)
					{
						LIBBMRM_MEMCPY(beta, beta_good, p3bmrm.nCP*sizeof(float64_t));
						LIBBMRM_MEMCPY(I2, I_good, p3bmrm.nCP*sizeof(uint32_t));
						alpha=alpha_good;
						qp_exitflag=qp_exitflag_good;
						flag=false;
					}
					else
					{
						if (alpha == 0)
						{
							alpha=1.0;
							alphaChanged=true;
						}
						else
						{
							alpha*=2;
							alphaChanged=true;
						}
					}
				}
				else
				{
					if (alpha > 0)
					{
						/* keep good solution and try for alpha /= 2 if previous alpha was 1 */
						LIBBMRM_MEMCPY(beta_good, beta, p3bmrm.nCP*sizeof(float64_t));
						LIBBMRM_MEMCPY(I_good, I2, p3bmrm.nCP*sizeof(uint32_t));
						alpha_good=alpha;
						qp_exitflag_good=qp_exitflag;
						isThereGoodSolution=true;

						if (alpha!=1.0)
						{
							alpha/=2.0;
							alphaChanged=true;
						}
						else
						{
							alpha=0.0;
							alphaChanged=true;
						}
					}
					else
					{
						flag=false;
					}
				}
			}
		}
		else
		{
			alphaChanged=false;
			LIBBMRM_MEMCPY(I2, I_start, p3bmrm.nCP*sizeof(uint32_t));
			LIBBMRM_MEMCPY(beta, beta_start, p3bmrm.nCP*sizeof(float64_t));

			/* add alpha-dependent terms to H, diag_h and b */
			cp_ptr=CPList_head;

			for (uint32_t i=0; i<p3bmrm.nCP; ++i)
			{
				rsum=0.0;
				A_1=get_cutting_plane(cp_ptr);
				cp_ptr=cp_ptr->next;

				for (uint32_t j=0; j<nDim; ++j)
					rsum+=A_1[j]*prevW[j];

				b2[i]=b[i]-((2*alpha)/(lambda+2*alpha))*rsum;
				diag_H2[i]=diag_H[i]/(lambda+2*alpha);

				for (uint32_t j=0; j<p3bmrm.nCP; ++j)
					H2[LIBBMRM_INDEX(i, j, BufSize)]=H[LIBBMRM_INDEX(i, j, BufSize)]/(lambda+2*alpha);
			}

			/* solve QP with current alpha */
			qp_exitflag=libqp_splx_solver(&get_col, diag_H2, b2, C, I2, S, beta,
					p3bmrm.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
			p3bmrm.qp_exitflag=qp_exitflag.exitflag;
			qp_cnt++;
		}
		/* ----------------------------------------------------------------------------------------------- */

		/* Update ICPcounter (add one to unused and reset used) + compute number of active CPs */
		p3bmrm.nzA=0;

		for (uint32_t aaa=0; aaa<p3bmrm.nCP; ++aaa)
		{
			if (beta[aaa]>epsilon)
			{
				++p3bmrm.nzA;
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

			for (uint32_t j=0; j<p3bmrm.nCP; ++j)
			{
				A_1=get_cutting_plane(cp_ptr);
				cp_ptr=cp_ptr->next;
				rsum+=A_1[i]*beta[j];
			}

			W[i]=(2*alpha*prevW[i]-rsum)/(lambda+2*alpha);
		}

		/* risk and subgradient computation */
		R=0.0;

		for (uint32_t p=0; p<cp_models; ++p)
		{
			Rt[p] = model->risk(subgrad_t[p], W, info[p]);
			b[p3bmrm.nCP+p]=-Rt[p];
			add_cutting_plane(&CPList_tail, map, A, find_free_idx(map, BufSize), subgrad_t[p], nDim);
			R+=Rt[p];
		}

		sq_norm_W=0.0;
		sq_norm_Wdiff=0.0;
		sq_norm_prevW=0.0;

		for (uint32_t j=0; j<nDim; ++j)
		{
			for (uint32_t p=0; p<cp_models; ++p)
				b[p3bmrm.nCP+p]+=subgrad_t[p][j]*W[j];

			sq_norm_W+=W[j]*W[j];
			sq_norm_Wdiff+=(W[j]-prevW[j])*(W[j]-prevW[j]);
			sq_norm_prevW+=prevW[j]*prevW[j];
		}

		/* compute Fp and Fd */
		p3bmrm.Fp=R+0.5*lambda*sq_norm_W + alpha*sq_norm_Wdiff;
		p3bmrm.Fd=-qp_exitflag.QP + ((alpha*lambda)/(lambda + 2*alpha))*sq_norm_prevW;

		/* gamma + tuneAlpha flag */
		if (alphaChanged)
		{
			eps=1.0-(p3bmrm.Fd/p3bmrm.Fp);
			gamma=(lastFp*(1-eps)-Fd_alpha0)/(Tmax*(1-eps));
		}

		if ((lastFp-p3bmrm.Fp) <= gamma)
		{
			tuneAlpha=true;
		}
		else
		{
			tuneAlpha=false;
		}

		/* Stopping conditions - set only with nonzero alpha */
		if (alpha==0.0)
		{
			if (p3bmrm.Fp-p3bmrm.Fd<=TolRel*LIBBMRM_ABS(p3bmrm.Fp))
				p3bmrm.exitflag=1;

			if (p3bmrm.Fp-p3bmrm.Fd<=TolAbs)
				p3bmrm.exitflag=2;
		}

		if (p3bmrm.nCP>=BufSize)
			p3bmrm.exitflag=-1;

		tstop=ttime.cur_time_diff(false);

		/* compute wdist (= || W_{t+1} - W_{t} || ) */
		sq_norm_Wdiff=0.0;

		for (uint32_t i=0; i<nDim; ++i)
		{
			sq_norm_Wdiff+=(W[i]-prevW[i])*(W[i]-prevW[i]);
		}

		wdist=::sqrt(sq_norm_Wdiff);

		/* Verbose output */
		if (verbose)
			SG_SPRINT("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, (Fp-Fd)=%lf, (Fp-Fd)/Fp=%lf, R=%lf, nCP=%d, nzA=%d, wdist=%lf, alpha=%lf, qp_cnt=%d, gamma=%lf, tuneAlpha=%d\n",
					p3bmrm.nIter, tstop-tstart, p3bmrm.Fp, p3bmrm.Fd, p3bmrm.Fp-p3bmrm.Fd,
					(p3bmrm.Fp-p3bmrm.Fd)/p3bmrm.Fp, R, p3bmrm.nCP, p3bmrm.nzA, wdist, alpha,
					qp_cnt, gamma, tuneAlpha);

		/* Check size of Buffer */
		if (p3bmrm.nCP>=BufSize)
		{
			p3bmrm.exitflag=-2;
			SG_SERROR("Buffer exceeded.\n");
		}

		/* keep w_t + Fp */
		LIBBMRM_MEMCPY(prevW, W, nDim*sizeof(float64_t));
		lastFp=p3bmrm.Fp;

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
				nCP_new=p3bmrm.nCP-cntICP;

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

					LIBBMRM_MEMMOVE(b+tmp_idx, b+tmp_idx+1, (p3bmrm.nCP+cp_models-tmp_idx)*sizeof(float64_t));
					LIBBMRM_MEMMOVE(beta+tmp_idx, beta+tmp_idx+1, (p3bmrm.nCP-tmp_idx)*sizeof(float64_t));
					LIBBMRM_MEMMOVE(diag_H+tmp_idx, diag_H+tmp_idx+1, (p3bmrm.nCP-tmp_idx)*sizeof(float64_t));
					LIBBMRM_MEMMOVE(I+tmp_idx, I+tmp_idx+1, (p3bmrm.nCP-tmp_idx)*sizeof(uint32_t));
					LIBBMRM_MEMMOVE(ICPcounter+tmp_idx, ICPcounter+tmp_idx+1, (p3bmrm.nCP-tmp_idx)*sizeof(uint32_t));
				}

				/* H */
				for (uint32_t i=0; i < nCP_new; ++i)
				{
					for (uint32_t j=0; j < nCP_new; ++j)
					{
						H_buff[LIBBMRM_INDEX(i, j, BufSize)]=H[LIBBMRM_INDEX(ACPs[i], ACPs[j], BufSize)];
					}
				}

				for (uint32_t i=0; i<nCP_new; ++i)
					for (uint32_t j=0; j<nCP_new; ++j)
						H[LIBBMRM_INDEX(i, j, BufSize)]=H_buff[LIBBMRM_INDEX(i, j, BufSize)];

				p3bmrm.nCP=nCP_new;
			}
		}
	} /* end of main loop */

	cp_ptr=CPList_head;

	while(cp_ptr != NULL)
	{
		cp_ptr2=cp_ptr;
		cp_ptr=cp_ptr->next;
		LIBBMRM_FREE(cp_ptr2);
	}

cleanup:

	LIBBMRM_FREE(H);
	LIBBMRM_FREE(b);
	LIBBMRM_FREE(beta);
	LIBBMRM_FREE(A);
	LIBBMRM_FREE(diag_H);
	LIBBMRM_FREE(I);
	LIBBMRM_FREE(ICPcounter);
	LIBBMRM_FREE(ICPs);
	LIBBMRM_FREE(ACPs);
	LIBBMRM_FREE(H_buff);
	LIBBMRM_FREE(map);
	LIBBMRM_FREE(prevW);
	LIBBMRM_FREE(wt);
	LIBBMRM_FREE(beta_start);
	LIBBMRM_FREE(beta_good);
	LIBBMRM_FREE(I_start);
	LIBBMRM_FREE(I_good);
	LIBBMRM_FREE(I2);
	LIBBMRM_FREE(b2);
	LIBBMRM_FREE(diag_H2);
	LIBBMRM_FREE(H2);
	LIBBMRM_FREE(C);
	LIBBMRM_FREE(S);

	for (uint32_t p=0; p<cp_models; ++p)
		LIBBMRM_FREE(subgrad_t[p]);

	LIBBMRM_FREE(subgrad_t);

	return(p3bmrm);
}
}
