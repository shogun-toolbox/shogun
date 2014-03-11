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

#include <shogun/structure/libbmrm.h>
#include <shogun/lib/external/libqp.h>
#include <shogun/lib/Time.h>
#include <shogun/io/SGIO.h>

#include <climits>
#include <limits>

namespace shogun
{
static const uint32_t QPSolverMaxIter=0xFFFFFFFF;
static const float64_t epsilon=0.0;

static float64_t *H;
uint32_t BufSize;

void add_cutting_plane(
		bmrm_ll**	tail,
		bool*		map,
		float64_t*	A,
		uint32_t	free_idx,
		float64_t*	cp_data,
		uint32_t	dim)
{
	REQUIRE(map[free_idx],
		"add_cutting_plane: CP index %u is not free\n", free_idx)

	LIBBMRM_MEMCPY(A+free_idx*dim, cp_data, dim*sizeof(float64_t));
	map[free_idx]=false;

	bmrm_ll *cp=(bmrm_ll*)LIBBMRM_CALLOC(1, bmrm_ll);

	if (cp==NULL)
	{
		SG_SERROR("Out of memory.\n")
		return;
	}

	/*
	MOD:
	cp->address=A+(free_idx*dim);
	cp->prev=*tail;
	cp->next=NULL;
	cp->idx=free_idx;*/
	cp->bmrm_ll_init(tail, NULL, A+(free_idx*dim), free_idx);
	(*tail)->next=cp;
	*tail=cp;
}

void remove_cutting_plane(
		bmrm_ll**	head,
		bmrm_ll**	tail,
		bool*		map,
		float64_t*	icp)
{
	bmrm_ll *cp_list_ptr=*head;

	//Removal of the cutting planes from head till that index
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

void clean_icp(ICP_stats* icp_stats,
		BmrmStatistics& bmrm,
		bmrm_ll** head,
		bmrm_ll** tail,
		float64_t*& Hmat,
		float64_t*& diag_H,
		float64_t*& beta,
		bool*& map,
		uint32_t cleanAfter,
		float64_t*& b,
		uint32_t*& I,
		uint32_t cp_models
		)
{
	/* find ICP */
	uint32_t cntICP=0;
	uint32_t cntACP=0;
	bmrm_ll* cp_ptr=*head;
	uint32_t tmp_idx=0;

	while (cp_ptr != *tail)
	{
		if (icp_stats->ICPcounter[tmp_idx++]>=cleanAfter)
		{
			icp_stats->ICPs[cntICP++]=cp_ptr->address;
		}
		else
		{
			icp_stats->ACPs[cntACP++]=tmp_idx-1;
		}

		cp_ptr=cp_ptr->next;
	}

	/* do ICP removal */
	if (cntICP > 0)
	{
		uint32_t nCP_new=bmrm.nCP-cntICP;

		for (uint32_t i=0; i<cntICP; ++i)
		{
			tmp_idx=0;
			cp_ptr=*head;

			while(cp_ptr->address != icp_stats->ICPs[i])
			{
				cp_ptr=cp_ptr->next;
				tmp_idx++;
			}

			remove_cutting_plane(head, tail, map, icp_stats->ICPs[i]);

			LIBBMRM_MEMMOVE(b+tmp_idx, b+tmp_idx+1,
					(bmrm.nCP+cp_models-tmp_idx)*sizeof(float64_t));
			LIBBMRM_MEMMOVE(beta+tmp_idx, beta+tmp_idx+1,
					(bmrm.nCP-tmp_idx)*sizeof(float64_t));
			LIBBMRM_MEMMOVE(diag_H+tmp_idx, diag_H+tmp_idx+1,
					(bmrm.nCP-tmp_idx)*sizeof(float64_t));
			LIBBMRM_MEMMOVE(I+tmp_idx, I+tmp_idx+1,
					(bmrm.nCP-tmp_idx)*sizeof(uint32_t));
			LIBBMRM_MEMMOVE(icp_stats->ICPcounter+tmp_idx, icp_stats->ICPcounter+tmp_idx+1,
					(bmrm.nCP-tmp_idx)*sizeof(uint32_t));
		}

		/* H */
		for (uint32_t i=0; i < nCP_new; ++i)
		{
			for (uint32_t j=0; j < nCP_new; ++j)
			{
				icp_stats->H_buff[LIBBMRM_INDEX(i, j, icp_stats->maxCPs)]=
					Hmat[LIBBMRM_INDEX(icp_stats->ACPs[i], icp_stats->ACPs[j], icp_stats->maxCPs)];
			}
		}

		for (uint32_t i=0; i<nCP_new; ++i)
			for (uint32_t j=0; j<nCP_new; ++j)
				Hmat[LIBBMRM_INDEX(i, j, icp_stats->maxCPs)]=
					icp_stats->H_buff[LIBBMRM_INDEX(i, j, icp_stats->maxCPs)];

		bmrm.nCP=nCP_new;
		ASSERT(bmrm.nCP<BufSize);
	}
}

/*----------------------------------------------------------------------
  Returns pointer at i-th column of Hessian matrix.
  ----------------------------------------------------------------------*/
static const float64_t *get_col( uint32_t i)
{
	return( &H[ BufSize*i ] );
}

BmrmStatistics svm_bmrm_solver(
		CDualLibQPBMSOSVM  *machine,
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
	BmrmStatistics bmrm;
	libqp_state_T qp_exitflag={0, 0, 0, 0};
	float64_t *b, *beta, *diag_H, *prevW;
	float64_t R, *subgrad, *A, QPSolverTolRel, C=1.0, wdist=0.0;
	floatmax_t rsum, sq_norm_W, sq_norm_Wdiff=0.0;
	uint32_t *I;
	uint8_t S=1;
	CStructuredModel* model=machine->get_model();
	uint32_t nDim=model->get_dim();
	CSOSVMHelper* helper = NULL;

	CTime ttime;
	float64_t tstart, tstop;

	bmrm_ll *CPList_head, *CPList_tail, *cp_ptr, *cp_ptr2, *cp_list=NULL;
	float64_t *A_1=NULL, *A_2=NULL;
	bool *map=NULL;


	tstart=ttime.cur_time_diff(false);

	BufSize=_BufSize;
	QPSolverTolRel=1e-9;

	uint32_t histSize = BufSize;
	H=NULL;
	b=NULL;
	beta=NULL;
	A=NULL;
	subgrad=NULL;
	diag_H=NULL;
	I=NULL;
	prevW=NULL;

	check_alloc(H, BufSize, 1);

	ASSERT(nDim > 0);
	ASSERT(BufSize > 0);
	REQUIRE(BufSize < (std::numeric_limits<size_t>::max() / nDim),
		"overflow: %u * %u > %u -- biggest possible BufSize=%u or nDim=%u\n",
		BufSize, nDim, std::numeric_limits<size_t>::max(),
		(std::numeric_limits<size_t>::max() / nDim),
		(std::numeric_limits<size_t>::max() / BufSize));

	check_alloc(A, BufSize, nDim);
	check_alloc(b, BufSize, 1);
	check_alloc(beta, BufSize, 1);
	check_alloc(subgrad, nDim, 1);
	check_alloc(diag_H, BufSize, 1);
	check_alloc(I, BufSize, 1);
	
	ICP_stats icp_stats;
	icp_stats.maxCPs = BufSize;
	check_alloc(icp_stats.ICPcounter, BufSize, 1);
	icp_stats.ICPs= (float64_t**) LIBBMRM_CALLOC(BufSize, float64_t*);
	if (icp_stats.ICPs==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	check_alloc(icp_stats.ACPs, BufSize, 1);

	/* Temporary buffers for ICP removal */
	icp_stats.H_buff= (float64_t*) LIBBMRM_CALLOC(BufSize*BufSize, float64_t);
	check_alloc(icp_stats.H_buff, BufSize, 1);

	map= (bool*) LIBBMRM_CALLOC(BufSize, bool);

	if (map==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	memset( (bool*) map, true, BufSize);

	cp_list= (bmrm_ll*) LIBBMRM_CALLOC(1, bmrm_ll);

	if (cp_list==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	check_alloc(prevW, nDim, 1);

	bmrm.hist_Fp = SGVector< float64_t >(histSize);
	bmrm.hist_Fd = SGVector< float64_t >(histSize);
	bmrm.hist_wdist = SGVector< float64_t >(histSize);

	/* Iinitial solution */
	R=machine->risk(subgrad, W);

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

	if (verbose)
		helper = machine->get_helper();

	/* main loop */
	ASSERT(bmrm.nCP<BufSize);
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
				rsum= SGVector<float64_t>::dot(A_1, A_2, nDim);

				H[LIBBMRM_INDEX(bmrm.nCP, i, BufSize)]
					= H[LIBBMRM_INDEX(i, bmrm.nCP, BufSize)]
					= rsum/_lambda;
			}
		}

		A_2=get_cutting_plane(CPList_tail);
		rsum = SGVector<float64_t>::dot(A_2, A_2, nDim);

		H[LIBBMRM_INDEX(bmrm.nCP, bmrm.nCP, BufSize)]=rsum/_lambda;

		diag_H[bmrm.nCP]=H[LIBBMRM_INDEX(bmrm.nCP, bmrm.nCP, BufSize)];
		I[bmrm.nCP]=1;

		beta[bmrm.nCP]=0.0; // [beta; 0]
		bmrm.nCP++;
		ASSERT(bmrm.nCP<BufSize);

#if 0
		/* TODO: scaling...*/
		float64_t scale = SGVector<float64_t>::max(diag_H, BufSize)/(1000.0*_lambda);
		SGVector<float64_t> sb(bmrm.nCP);
		sb.zero();
		sb.vec1_plus_scalar_times_vec2(sb.vector, 1/scale, b, bmrm.nCP);

		SGVector<float64_t> sh(bmrm.nCP);
		sh.zero();
		sb.vec1_plus_scalar_times_vec2(sh.vector, 1/scale, diag_H, bmrm.nCP);

		qp_exitflag =
			libqp_splx_solver(&get_col, sh.vector, sb.vector, &C, I, &S, beta,
				bmrm.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
#else
		/* call QP solver */
		qp_exitflag=libqp_splx_solver(&get_col, diag_H, b, &C, I, &S, beta,
				bmrm.nCP, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBBMRM_PLUS_INF, 0);
#endif

		bmrm.qp_exitflag=qp_exitflag.exitflag;

		/* Update ICPcounter (add one to unused and reset used)
		 * + compute number of active CPs */
		bmrm.nzA=0;

		for (uint32_t aaa=0; aaa<bmrm.nCP; ++aaa)
		{
			if (beta[aaa]>epsilon)
			{
				++bmrm.nzA;
				icp_stats.ICPcounter[aaa]=0;
			}
			else
			{
				icp_stats.ICPcounter[aaa]+=1;
			}
		}

		/* W update */
		memset(W, 0, sizeof(float64_t)*nDim);
		cp_ptr=CPList_head;
		for (uint32_t j=0; j<bmrm.nCP; ++j)
		{
			A_1=get_cutting_plane(cp_ptr);
			cp_ptr=cp_ptr->next;
			SGVector<float64_t>::vec1_plus_scalar_times_vec2(W, -beta[j]/_lambda, A_1, nDim);
		}

		/* risk and subgradient computation */
		R = machine->risk(subgrad, W);
		add_cutting_plane(&CPList_tail, map, A,
				find_free_idx(map, BufSize), subgrad, nDim);

		sq_norm_W=SGVector<float64_t>::dot(W, W, nDim);
		b[bmrm.nCP]=SGVector<float64_t>::dot(subgrad, W, nDim) - R;

		sq_norm_Wdiff=0.0;
		for (uint32_t j=0; j<nDim; ++j)
		{
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

		tstop=ttime.cur_time_diff(false);

		/* Verbose output */
		if (verbose)
			SG_SPRINT("%4d: tim=%.3lf, Fp=%lf, Fd=%lf, (Fp-Fd)=%lf, (Fp-Fd)/Fp=%lf, R=%lf, nCP=%d, nzA=%d, QPexitflag=%d\n",
					bmrm.nIter, tstop-tstart, bmrm.Fp, bmrm.Fd, bmrm.Fp-bmrm.Fd,
					(bmrm.Fp-bmrm.Fd)/bmrm.Fp, R, bmrm.nCP, bmrm.nzA, qp_exitflag.exitflag);

		// iteration exceeds histSize
		if (bmrm.nIter >= histSize)
		{
			histSize += BufSize;
			bmrm.hist_Fp.resize_vector(histSize);
			bmrm.hist_Fd.resize_vector(histSize);
			bmrm.hist_wdist.resize_vector(histSize);
		}

		/* Keep Fp, Fd and w_dist history */
		ASSERT(bmrm.nIter < histSize);
		bmrm.hist_Fp[bmrm.nIter]=bmrm.Fp;
		bmrm.hist_Fd[bmrm.nIter]=bmrm.Fd;
		bmrm.hist_wdist[bmrm.nIter]=wdist;

		/* keep W (for wdist history track) */
		LIBBMRM_MEMCPY(prevW, W, nDim*sizeof(float64_t));

		/* Inactive Cutting Planes (ICP) removal */
		if (cleanICP)
		{
			clean_icp(&icp_stats, bmrm, &CPList_head, &CPList_tail, H, diag_H, beta, map, cleanAfter, b, I);
			ASSERT(bmrm.nCP<BufSize);
		}

		// next CP would exceed BufSize
		if (bmrm.nCP+1 >= BufSize)
			bmrm.exitflag=-1;

		/* Debug: compute objective and training error */
		if (verbose && SG_UNLIKELY(sg_io->loglevel_above(MSG_DEBUG)))
		{
			float64_t debug_tstart=ttime.cur_time_diff(false);

			SGVector<float64_t> w_debug(W, nDim, false);
			float64_t primal = CSOSVMHelper::primal_objective(w_debug, model, _lambda);
			float64_t train_error = CSOSVMHelper::average_loss(w_debug, model);
			helper->add_debug_info(primal, bmrm.nIter, train_error);

			float64_t debug_tstop=ttime.cur_time_diff(false);
			SG_SPRINT("%4d: add_debug_info: tim=%.3lf, primal=%.3lf, train_error=%lf\n",
				bmrm.nIter, debug_tstop-debug_tstart, primal, train_error);
		}

	} /* end of main loop */

	if (verbose)
	{
		helper->terminate();
		SG_UNREF(helper);
	}

	ASSERT(bmrm.nIter+1 <= histSize);
	bmrm.hist_Fp.resize_vector(bmrm.nIter+1);
	bmrm.hist_Fd.resize_vector(bmrm.nIter+1);
	bmrm.hist_wdist.resize_vector(bmrm.nIter+1);

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
	LIBBMRM_FREE(icp_stats.ICPcounter);
	LIBBMRM_FREE(icp_stats.ICPs);
	LIBBMRM_FREE(icp_stats.ACPs);
	LIBBMRM_FREE(icp_stats.H_buff);
	LIBBMRM_FREE(map);
	LIBBMRM_FREE(prevW);

	if (cp_list)
		LIBBMRM_FREE(cp_list);

	SG_UNREF(model);

	return(bmrm);
}
}
