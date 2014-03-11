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

	cp->bmrm_ll_init(*tail, NULL, A+(free_idx*dim));
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


	H= (float64_t*) LIBBMRM_CALLOC(BufSize*BufSize, float64_t);

	if (H==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	ASSERT(nDim > 0);
	ASSERT(BufSize > 0);
	REQUIRE(BufSize < (std::numeric_limits<size_t>::max() / nDim),
		"overflow: %u * %u > %u -- biggest possible BufSize=%u or nDim=%u\n",
		BufSize, nDim, std::numeric_limits<size_t>::max(),
		(std::numeric_limits<size_t>::max() / nDim),
		(std::numeric_limits<size_t>::max() / BufSize));

	A= (float64_t*) LIBBMRM_CALLOC(size_t(nDim)*size_t(BufSize), float64_t);

	if (A==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	b= (float64_t*) LIBBMRM_CALLOC(BufSize, float64_t);

	if (b==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	beta= (float64_t*) LIBBMRM_CALLOC(BufSize, float64_t);

	if (beta==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	subgrad= (float64_t*) LIBBMRM_CALLOC(nDim, float64_t);

	if (subgrad==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	diag_H= (float64_t*) LIBBMRM_CALLOC(BufSize, float64_t);

	if (diag_H==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	I= (uint32_t*) LIBBMRM_CALLOC(BufSize, uint32_t);

	if (I==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	ICP_stats icp_stats;
	icp_stats.maxCPs = BufSize;

	icp_stats.ICPcounter= (uint32_t*) LIBBMRM_CALLOC(BufSize, uint32_t);
	if (icp_stats.ICPcounter==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	icp_stats.ICPs= (float64_t**) LIBBMRM_CALLOC(BufSize, float64_t*);
	if (icp_stats.ICPs==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	icp_stats.ACPs= (uint32_t*) LIBBMRM_CALLOC(BufSize, uint32_t);
	if (icp_stats.ACPs==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	/* Temporary buffers for ICP removal */
	icp_stats.H_buff= (float64_t*) LIBBMRM_CALLOC(BufSize*BufSize, float64_t);
	if (icp_stats.H_buff==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	map= (bool*) LIBBMRM_CALLOC(BufSize, bool);

	if (map==NULL)
	{
		bmrm.exitflag=-2;
		goto cleanup;
	}

	memset( (bool*) map, 
