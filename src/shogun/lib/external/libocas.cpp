/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * libocas.c: Implementation of the OCAS solver for training
 *            linear SVM classifiers.
 *
 * Copyright (C) 2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 *                    Soeren Sonnenburg, soeren.sonnenburg@first.fraunhofer.de
 *-------------------------------------------------------------------- */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>

#include <shogun/lib/external/libocas.h>
#include <shogun/lib/external/libocas_common.h>
#include <shogun/lib/external/libqp.h>

namespace shogun
{
#define MU 0.1      /* must be from (0,1>   1..means that OCAS becomes equivalent to CPA */
                    /* see paper Franc&Sonneburg JMLR 2009 */

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

/*----------------------------------------------------------------------
  Returns time of the day in seconds.
  ----------------------------------------------------------------------*/
static float64_t get_time()
{
	struct timeval tv;
	if (gettimeofday(&tv, NULL)==0)
		return tv.tv_sec+((float64_t)(tv.tv_usec))/1e6;
	else
		return 0.0;
}


/*----------------------------------------------------------------------
  Linear binary Ocas-SVM solver with additinal contraint enforceing
  a subset of weights (indices of the weights given by num_nnw/nnw_idx)
  to be non-negative.

  ----------------------------------------------------------------------*/
ocas_return_value_T svm_ocas_solver_nnw(
            float64_t C,
            uint32_t nData,
            uint32_t num_nnw,
            uint32_t* nnw_idx,
            float64_t TolRel,
            float64_t TolAbs,
            float64_t QPBound,
            float64_t MaxTime,
            uint32_t _BufSize,
            uint8_t Method,
            int (*add_pw_constr)(uint32_t, uint32_t, void*),
            void (*clip_neg_W)(uint32_t, uint32_t*, void*),
            void (*compute_W)(float64_t*, float64_t*, float64_t*, uint32_t, void*),
            float64_t (*update_W)(float64_t, void*),
            int (*add_new_cut)(float64_t*, uint32_t*, uint32_t, uint32_t, void*),
            int (*compute_output)(float64_t*, void* ),
            int (*sort)(float64_t*, float64_t*, uint32_t),
                        void (*ocas_print)(ocas_return_value_T),
                        void* user_data)
{
  ocas_return_value_T ocas={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  float64_t *b, *alpha, *diag_H;
  float64_t *output, *old_output;
  float64_t xi, sq_norm_W, QPSolverTolRel, dot_prod_WoldW, sq_norm_oldW;
  float64_t A0, B0, GradVal, t, t1, t2, *Ci, *Bi, *hpf, *hpb;
  float64_t start_time, ocas_start_time;
  uint32_t cut_length;
  uint32_t i, *new_cut;
  uint32_t *I;
/*  uint8_t S = 1; */
  libqp_state_T qp_exitflag;

  float64_t max_cp_norm;
  float64_t max_b;
  float64_t Cs[2];
  uint8_t S[2];

  ocas_start_time = get_time();
  ocas.qp_solver_time = 0;
  ocas.output_time = 0;
  ocas.sort_time = 0;
  ocas.add_time = 0;
  ocas.w_time = 0;
  ocas.print_time = 0;

  BufSize = _BufSize;

  QPSolverTolRel = TolRel*0.5;

  H=NULL;
  b=NULL;
  alpha=NULL;
  new_cut=NULL;
  I=NULL;
  diag_H=NULL;
  output=NULL;
  old_output=NULL;
  hpf=NULL;
  hpb = NULL;
  Ci=NULL;
  Bi=NULL;

  /* Hessian matrix contains dot product of normal vectors of selected cutting planes */
  H = (float64_t*)LIBOCAS_CALLOC(BufSize*BufSize, float64_t);
  if(H == NULL)
  {
          ocas.exitflag=-2;
          goto cleanup;
  }

  /* bias of cutting planes */
  b = (float64_t*)LIBOCAS_CALLOC(BufSize, float64_t);
  if(b == NULL)
  {
          ocas.exitflag=-2;
          goto cleanup;
  }

  alpha = (float64_t*)LIBOCAS_CALLOC(BufSize, float64_t);
  if(alpha == NULL)
  {
          ocas.exitflag=-2;
          goto cleanup;
  }

  /* indices of examples which define a new cut */
  new_cut = (uint32_t*)LIBOCAS_CALLOC(nData, uint32_t);
  if(new_cut == NULL)
  {
          ocas.exitflag=-2;
          goto cleanup;
  }

  I = (uint32_t*)LIBOCAS_CALLOC(BufSize, uint32_t);
  if(I == NULL)
  {
          ocas.exitflag=-2;
          goto cleanup;
  }

  for(i=0; i< BufSize; i++) I[i] = 2;

  diag_H = (float64_t*)LIBOCAS_CALLOC(BufSize, float64_t);
  if(diag_H == NULL)
  {
          ocas.exitflag=-2;
          goto cleanup;
  }

  output = (float64_t*)LIBOCAS_CALLOC(nData, float64_t);
  if(output == NULL)
  {
          ocas.exitflag=-2;
          goto cleanup;
  }

  old_output = (float64_t*)LIBOCAS_CALLOC(nData, float64_t);
  if(old_output == NULL)
  {
          ocas.exitflag=-2;
          goto cleanup;
  }

  /* array of hinge points used in line-serach  */
  hpf = (float64_t*) LIBOCAS_CALLOC(nData, float64_t);
  if(hpf == NULL)
  {
          ocas.exitflag=-2;
          goto cleanup;
  }

  hpb = (float64_t*) LIBOCAS_CALLOC(nData, float64_t);
  if(hpb == NULL)
  {
          ocas.exitflag=-2;
          goto cleanup;
  }

  /* vectors Ci, Bi are used in the line search procedure */
  Ci = (float64_t*)LIBOCAS_CALLOC(nData, float64_t);
  if(Ci == NULL)
  {
          ocas.exitflag=-2;
          goto cleanup;
  }

  Bi = (float64_t*)LIBOCAS_CALLOC(nData, float64_t);
  if(Bi == NULL)
  {
          ocas.exitflag=-2;
          goto cleanup;
  }

  /* initial cutting planes implementing the non-negativity constraints on W*/
  Cs[0]=10000000.0;
  Cs[1]=C;
  S[0]=1;
  S[1]=1;
  for(i=0; i < num_nnw; i++)
  {
      if(add_pw_constr(nnw_idx[i],i, user_data) != 0)
      {
          ocas.exitflag=-2;
          goto cleanup;
      }
      diag_H[i] = 1.0;
      H[LIBOCAS_INDEX(i,i,BufSize)] = 1.0;
      I[i] = 1;
  }

  max_cp_norm = 1;
  max_b = 0;

  /*  */
  ocas.nCutPlanes = num_nnw;
  ocas.exitflag = 0;
  ocas.nIter = 0;

  /* Compute initial value of Q_P assuming that W is zero vector.*/
  sq_norm_W = 0;
  xi = nData;
  ocas.Q_P = 0.5*sq_norm_W + C*xi;
  ocas.Q_D = 0;

  /* Compute the initial cutting plane */
  cut_length = nData;
  for(i=0; i < nData; i++)
    new_cut[i] = i;

  ocas.trn_err = nData;
  ocas.ocas_time = get_time() - ocas_start_time;
  /*  ocas_print("%4d: tim=%f, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, Q_P-Q_D/abs(Q_P)=%f\n",
          ocas.nIter,cur_time, ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/LIBOCAS_ABS(ocas.Q_P));
  */
  ocas_print(ocas);

  /* main loop */
  while( ocas.exitflag == 0 )
  {
    ocas.nIter++;

    /* append a new cut to the buffer and update H */
    b[ocas.nCutPlanes] = -(float64_t)cut_length;

    max_b = LIBOCAS_MAX(max_b,(float64_t)cut_length);

    start_time = get_time();

    if(add_new_cut( &H[LIBOCAS_INDEX(0,ocas.nCutPlanes,BufSize)], new_cut, cut_length, ocas.nCutPlanes, user_data ) != 0)
    {
          ocas.exitflag=-2;
          goto cleanup;
    }

    ocas.add_time += get_time() - start_time;

    /* copy new added row:  H(ocas.nCutPlanes,ocas.nCutPlanes,1:ocas.nCutPlanes-1) = H(1:ocas.nCutPlanes-1:ocas.nCutPlanes)' */
    diag_H[ocas.nCutPlanes] = H[LIBOCAS_INDEX(ocas.nCutPlanes,ocas.nCutPlanes,BufSize)];
    for(i=0; i < ocas.nCutPlanes; i++) {
      H[LIBOCAS_INDEX(ocas.nCutPlanes,i,BufSize)] = H[LIBOCAS_INDEX(i,ocas.nCutPlanes,BufSize)];
    }

    max_cp_norm = LIBOCAS_MAX(max_cp_norm, sqrt(diag_H[ocas.nCutPlanes]));


    ocas.nCutPlanes++;

    /* call inner QP solver */
    start_time = get_time();

    /* compute upper bound on sum of dual variables associated with the positivity constraints */
    Cs[0] = sqrt((float64_t)nData)*(sqrt(C)*sqrt(max_b) + C*max_cp_norm);

/*    qp_exitflag = libqp_splx_solver(&get_col, diag_H, b, &C, I, &S, alpha,
                                  ocas.nCutPlanes, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBOCAS_PLUS_INF,0);*/
    qp_exitflag = libqp_splx_solver(&get_col, diag_H, b, Cs, I, S, alpha,
                                  ocas.nCutPlanes, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBOCAS_PLUS_INF,0);

    ocas.qp_exitflag = qp_exitflag.exitflag;

    ocas.qp_solver_time += get_time() - start_time;
    ocas.Q_D = -qp_exitflag.QP;

    ocas.nNZAlpha = 0;
    for(i=0; i < ocas.nCutPlanes; i++) {
      if( alpha[i] != 0) ocas.nNZAlpha++;
    }

    sq_norm_oldW = sq_norm_W;
    start_time = get_time();
    compute_W( &sq_norm_W, &dot_prod_WoldW, alpha, ocas.nCutPlanes, user_data );
    clip_neg_W(num_nnw, nnw_idx, user_data);
    ocas.w_time += get_time() - start_time;

    /* select a new cut */
    switch( Method )
    {
      /* cutting plane algorithm implemented in SVMperf and BMRM */
      case 0:

        start_time = get_time();
        if( compute_output( output, user_data ) != 0)
        {
          ocas.exitflag=-2;
          goto cleanup;
        }
        ocas.output_time += get_time()-start_time;

        xi = 0;
        cut_length = 0;
        ocas.trn_err = 0;
        for(i=0; i < nData; i++)
        {
          if(output[i] <= 0) ocas.trn_err++;

          if(output[i] <= 1) {
            xi += 1 - output[i];
            new_cut[cut_length] = i;
            cut_length++;
          }
        }
        ocas.Q_P = 0.5*sq_norm_W + C*xi;

        ocas.ocas_time = get_time() - ocas_start_time;

        /*        ocas_print("%4d: tim=%f, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, 1-Q_D/Q_P=%f, nza=%4d, err=%.2f%%, qpf=%d\n",
                  ocas.nIter,cur_time, ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/LIBOCAS_ABS(ocas.Q_P),
                  ocas.nNZAlpha, 100*(float64_t)ocas.trn_err/(float64_t)nData, ocas.qp_exitflag );
        */

        start_time = get_time();
        ocas_print(ocas);
        ocas.print_time += get_time() - start_time;

        break;


      /* Ocas strategy */
      case 1:

        /* Linesearch */
        A0 = sq_norm_W -2*dot_prod_WoldW + sq_norm_oldW;
        B0 = dot_prod_WoldW - sq_norm_oldW;

        memcpy( old_output, output, sizeof(float64_t)*nData );

        start_time = get_time();
        if( compute_output( output, user_data ) != 0)
        {
          ocas.exitflag=-2;
          goto cleanup;
        }
        ocas.output_time += get_time()-start_time;

        uint32_t num_hp = 0;
        GradVal = B0;
        for(i=0; i< nData; i++) {

          Ci[i] = C*(1-old_output[i]);
          Bi[i] = C*(old_output[i] - output[i]);

          float64_t val;
          if(Bi[i] != 0)
            val = -Ci[i]/Bi[i];
          else
            val = -LIBOCAS_PLUS_INF;

          if (val>0)
          {
/*            hpi[num_hp] = i;*/
            hpb[num_hp] = Bi[i];
            hpf[num_hp] = val;
            num_hp++;
          }

          if( (Bi[i] < 0 && val > 0) || (Bi[i] > 0 && val <= 0))
            GradVal += Bi[i];

        }

        t = 0;
        if( GradVal < 0 )
        {
          start_time = get_time();
/*          if( sort(hpf, hpi, num_hp) != 0)*/
          if( sort(hpf, hpb, num_hp) != 0 )
          {
            ocas.exitflag=-2;
            goto cleanup;
          }
          ocas.sort_time += get_time() - start_time;

          float64_t t_new, GradVal_new;
          i = 0;
          while( GradVal < 0 && i < num_hp )
          {
            t_new = hpf[i];
            GradVal_new = GradVal + LIBOCAS_ABS(hpb[i]) + A0*(t_new-t);

            if( GradVal_new >= 0 )
            {
              t = t + GradVal*(t-t_new)/(GradVal_new - GradVal);
            }
            else
            {
              t = t_new;
              i++;
            }

            GradVal = GradVal_new;
          }
        }

        /*
        t = hpf[0] - 1;
        i = 0;
        GradVal = t*A0 + Bsum;
        while( GradVal < 0 && i < num_hp && hpf[i] < LIBOCAS_PLUS_INF ) {
          t = hpf[i];
          Bsum = Bsum + LIBOCAS_ABS(Bi[hpi[i]]);
          GradVal = t*A0 + Bsum;
          i++;
        }
        */
        t = LIBOCAS_MAX(t,0);          /* just sanity check; t < 0 should not ocure */

        /* this guarantees that the new solution will not violate the positivity constraints on W */
        t = LIBOCAS_MIN(t,1);

        t1 = t;                /* new (best so far) W */
        t2 = t+MU*(1.0-t);   /* new cutting plane */
        /*        t2 = t+(1.0-t)/10.0;   */

        /* update W to be the best so far solution */
        sq_norm_W = update_W( t1, user_data );

        /* select a new cut */
        xi = 0;
        cut_length = 0;
        ocas.trn_err = 0;
        for(i=0; i < nData; i++ ) {

          if( (old_output[i]*(1-t2) + t2*output[i]) <= 1 )
          {
            new_cut[cut_length] = i;
            cut_length++;
          }

          output[i] = old_output[i]*(1-t1) + t1*output[i];

          if( output[i] <= 1) xi += 1-output[i];
          if( output[i] <= 0) ocas.trn_err++;

        }

        ocas.Q_P = 0.5*sq_norm_W + C*xi;

        ocas.ocas_time = get_time() - ocas_start_time;

        /*        ocas_print("%4d: tim=%f, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, 1-Q_D/Q_P=%f, nza=%4d, err=%.2f%%, qpf=%d\n",
                   ocas.nIter, cur_time, ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/LIBOCAS_ABS(ocas.Q_P),
                   ocas.nNZAlpha, 100*(float64_t)ocas.trn_err/(float64_t)nData, ocas.qp_exitflag );
        */

        start_time = get_time();
        ocas_print(ocas);
        ocas.print_time += get_time() - start_time;

        break;
    }

    /* Stopping conditions */
    if( ocas.Q_P - ocas.Q_D <= TolRel*LIBOCAS_ABS(ocas.Q_P)) ocas.exitflag = 1;
    if( ocas.Q_P - ocas.Q_D <= TolAbs) ocas.exitflag = 2;
    if( ocas.Q_P <= QPBound) ocas.exitflag = 3;
    if( MaxTime > 0 && ocas.ocas_time >= MaxTime) ocas.exitflag = 4;
    if(ocas.nCutPlanes >= BufSize) ocas.exitflag = -1;

  } /* end of the main loop */

cleanup:

  LIBOCAS_FREE(H);
  LIBOCAS_FREE(b);
  LIBOCAS_FREE(alpha);
  LIBOCAS_FREE(new_cut);
  LIBOCAS_FREE(I);
  LIBOCAS_FREE(diag_H);
  LIBOCAS_FREE(output);
  LIBOCAS_FREE(old_output);
  LIBOCAS_FREE(hpf);
/*  LIBOCAS_FREE(hpi);*/
  LIBOCAS_FREE(hpb);
  LIBOCAS_FREE(Ci);
  LIBOCAS_FREE(Bi);

  ocas.ocas_time = get_time() - ocas_start_time;

  return(ocas);
}



/*----------------------------------------------------------------------
  Linear binary Ocas-SVM solver.
  ----------------------------------------------------------------------*/
ocas_return_value_T svm_ocas_solver(
            float64_t C,
            uint32_t nData,
            float64_t TolRel,
            float64_t TolAbs,
            float64_t QPBound,
            float64_t MaxTime,
            uint32_t _BufSize,
            uint8_t Method,
            void (*compute_W)(float64_t*, float64_t*, float64_t*, uint32_t, void*),
            float64_t (*update_W)(float64_t, void*),
            int (*add_new_cut)(float64_t*, uint32_t*, uint32_t, uint32_t, void*),
            int (*compute_output)(float64_t*, void* ),
            int (*sort)(float64_t*, float64_t*, uint32_t),
			void (*ocas_print)(ocas_return_value_T),
			void* user_data)
{
  ocas_return_value_T ocas={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  float64_t *b, *alpha, *diag_H;
  float64_t *output, *old_output;
  float64_t xi, sq_norm_W, QPSolverTolRel, dot_prod_WoldW, sq_norm_oldW;
  float64_t A0, B0, GradVal, t, t1, t2, *Ci, *Bi, *hpf, *hpb;
  float64_t start_time, ocas_start_time;
  uint32_t cut_length;
  uint32_t i, *new_cut;
  uint32_t *I;
  uint8_t S = 1;
  libqp_state_T qp_exitflag;

  ocas_start_time = get_time();
  ocas.qp_solver_time = 0;
  ocas.output_time = 0;
  ocas.sort_time = 0;
  ocas.add_time = 0;
  ocas.w_time = 0;
  ocas.print_time = 0;
  float64_t gap;

  BufSize = _BufSize;

  QPSolverTolRel = TolRel*0.5;

  H=NULL;
  b=NULL;
  alpha=NULL;
  new_cut=NULL;
  I=NULL;
  diag_H=NULL;
  output=NULL;
  old_output=NULL;
  hpf=NULL;
  hpb = NULL;
  Ci=NULL;
  Bi=NULL;

  /* Hessian matrix contains dot product of normal vectors of selected cutting planes */
  H = (float64_t*)LIBOCAS_CALLOC(BufSize*BufSize, float64_t);
  if(H == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* bias of cutting planes */
  b = (float64_t*)LIBOCAS_CALLOC(BufSize, float64_t);
  if(b == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  alpha = (float64_t*)LIBOCAS_CALLOC(BufSize, float64_t);
  if(alpha == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* indices of examples which define a new cut */
  new_cut = (uint32_t*)LIBOCAS_CALLOC(nData, uint32_t);
  if(new_cut == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  I = (uint32_t*)LIBOCAS_CALLOC(BufSize, uint32_t);
  if(I == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  for(i=0; i< BufSize; i++) I[i] = 1;

  diag_H = (float64_t*)LIBOCAS_CALLOC(BufSize, float64_t);
  if(diag_H == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  output = (float64_t*)LIBOCAS_CALLOC(nData, float64_t);
  if(output == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  old_output = (float64_t*)LIBOCAS_CALLOC(nData, float64_t);
  if(old_output == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* array of hinge points used in line-serach  */
  hpf = (float64_t*) LIBOCAS_CALLOC(nData, float64_t);
  if(hpf == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  hpb = (float64_t*) LIBOCAS_CALLOC(nData, float64_t);
  if(hpb == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* vectors Ci, Bi are used in the line search procedure */
  Ci = (float64_t*)LIBOCAS_CALLOC(nData, float64_t);
  if(Ci == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  Bi = (float64_t*)LIBOCAS_CALLOC(nData, float64_t);
  if(Bi == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  ocas.nCutPlanes = 0;
  ocas.exitflag = 0;
  ocas.nIter = 0;

  /* Compute initial value of Q_P assuming that W is zero vector.*/
  sq_norm_W = 0;
  xi = nData;
  ocas.Q_P = 0.5*sq_norm_W + C*xi;
  ocas.Q_D = 0;

  /* Compute the initial cutting plane */
  cut_length = nData;
  for(i=0; i < nData; i++)
    new_cut[i] = i;

	gap=(ocas.Q_P-ocas.Q_D)/CMath::abs(ocas.Q_P);
	SG_SABS_PROGRESS(gap, -CMath::log10(gap), -CMath::log10(1), -CMath::log10(TolRel), 6)

  ocas.trn_err = nData;
  ocas.ocas_time = get_time() - ocas_start_time;
  /*  ocas_print("%4d: tim=%f, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, Q_P-Q_D/abs(Q_P)=%f\n",
          ocas.nIter,cur_time, ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/LIBOCAS_ABS(ocas.Q_P));
  */
  ocas_print(ocas);

  /* main loop */
  while( ocas.exitflag == 0 )
  {
    ocas.nIter++;

    /* append a new cut to the buffer and update H */
    b[ocas.nCutPlanes] = -(float64_t)cut_length;

    start_time = get_time();

    if(add_new_cut( &H[LIBOCAS_INDEX(0,ocas.nCutPlanes,BufSize)], new_cut, cut_length, ocas.nCutPlanes, user_data ) != 0)
    {
	  ocas.exitflag=-2;
	  goto cleanup;
    }

    ocas.add_time += get_time() - start_time;

    /* copy new added row:  H(ocas.nCutPlanes,ocas.nCutPlanes,1:ocas.nCutPlanes-1) = H(1:ocas.nCutPlanes-1:ocas.nCutPlanes)' */
    diag_H[ocas.nCutPlanes] = H[LIBOCAS_INDEX(ocas.nCutPlanes,ocas.nCutPlanes,BufSize)];
    for(i=0; i < ocas.nCutPlanes; i++) {
      H[LIBOCAS_INDEX(ocas.nCutPlanes,i,BufSize)] = H[LIBOCAS_INDEX(i,ocas.nCutPlanes,BufSize)];
    }

    ocas.nCutPlanes++;

    /* call inner QP solver */
    start_time = get_time();

    qp_exitflag = libqp_splx_solver(&get_col, diag_H, b, &C, I, &S, alpha,
                                  ocas.nCutPlanes, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBOCAS_PLUS_INF,0);

    ocas.qp_exitflag = qp_exitflag.exitflag;

    ocas.qp_solver_time += get_time() - start_time;
    ocas.Q_D = -qp_exitflag.QP;

    ocas.nNZAlpha = 0;
    for(i=0; i < ocas.nCutPlanes; i++) {
      if( alpha[i] != 0) ocas.nNZAlpha++;
    }

    sq_norm_oldW = sq_norm_W;
    start_time = get_time();
    compute_W( &sq_norm_W, &dot_prod_WoldW, alpha, ocas.nCutPlanes, user_data );
    ocas.w_time += get_time() - start_time;

    /* select a new cut */
    switch( Method )
    {
      /* cutting plane algorithm implemented in SVMperf and BMRM */
      case 0:

        start_time = get_time();
        if( compute_output( output, user_data ) != 0)
        {
          ocas.exitflag=-2;
          goto cleanup;
        }
        ocas.output_time += get_time()-start_time;
				gap=(ocas.Q_P-ocas.Q_D)/CMath::abs(ocas.Q_P);
        SG_SABS_PROGRESS(gap, -CMath::log10(gap), -CMath::log10(1), -CMath::log10(TolRel), 6)

        xi = 0;
        cut_length = 0;
        ocas.trn_err = 0;
        for(i=0; i < nData; i++)
        {
          if(output[i] <= 0) ocas.trn_err++;

          if(output[i] <= 1) {
            xi += 1 - output[i];
            new_cut[cut_length] = i;
            cut_length++;
          }
        }
        ocas.Q_P = 0.5*sq_norm_W + C*xi;

        ocas.ocas_time = get_time() - ocas_start_time;

        /*        ocas_print("%4d: tim=%f, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, 1-Q_D/Q_P=%f, nza=%4d, err=%.2f%%, qpf=%d\n",
                  ocas.nIter,cur_time, ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/LIBOCAS_ABS(ocas.Q_P),
                  ocas.nNZAlpha, 100*(float64_t)ocas.trn_err/(float64_t)nData, ocas.qp_exitflag );
        */

        start_time = get_time();
        ocas_print(ocas);
        ocas.print_time += get_time() - start_time;

        break;


      /* Ocas strategy */
      case 1:

        /* Linesearch */
        A0 = sq_norm_W -2*dot_prod_WoldW + sq_norm_oldW;
        B0 = dot_prod_WoldW - sq_norm_oldW;

        memcpy( old_output, output, sizeof(float64_t)*nData );

        start_time = get_time();
        if( compute_output( output, user_data ) != 0)
        {
          ocas.exitflag=-2;
          goto cleanup;
        }
        ocas.output_time += get_time()-start_time;

        uint32_t num_hp = 0;
        GradVal = B0;
        for(i=0; i< nData; i++) {

          Ci[i] = C*(1-old_output[i]);
          Bi[i] = C*(old_output[i] - output[i]);

          float64_t val;
          if(Bi[i] != 0)
            val = -Ci[i]/Bi[i];
          else
            val = -LIBOCAS_PLUS_INF;

          if (val>0)
          {
/*            hpi[num_hp] = i;*/
            hpb[num_hp] = Bi[i];
            hpf[num_hp] = val;
            num_hp++;
          }

          if( (Bi[i] < 0 && val > 0) || (Bi[i] > 0 && val <= 0))
            GradVal += Bi[i];

        }

        t = 0;
        if( GradVal < 0 )
        {
          start_time = get_time();
/*          if( sort(hpf, hpi, num_hp) != 0)*/
          if( sort(hpf, hpb, num_hp) != 0 )
          {
            ocas.exitflag=-2;
            goto cleanup;
          }
          ocas.sort_time += get_time() - start_time;

          float64_t t_new, GradVal_new;
          i = 0;
          while( GradVal < 0 && i < num_hp )
          {
            t_new = hpf[i];
            GradVal_new = GradVal + LIBOCAS_ABS(hpb[i]) + A0*(t_new-t);

            if( GradVal_new >= 0 )
            {
              t = t + GradVal*(t-t_new)/(GradVal_new - GradVal);
            }
            else
            {
              t = t_new;
              i++;
            }

            GradVal = GradVal_new;
          }
        }

        /*
        t = hpf[0] - 1;
        i = 0;
        GradVal = t*A0 + Bsum;
        while( GradVal < 0 && i < num_hp && hpf[i] < LIBOCAS_PLUS_INF ) {
          t = hpf[i];
          Bsum = Bsum + LIBOCAS_ABS(Bi[hpi[i]]);
          GradVal = t*A0 + Bsum;
          i++;
        }
        */
        t = LIBOCAS_MAX(t,0);          /* just sanity check; t < 0 should not ocure */

        t1 = t;                /* new (best so far) W */
        t2 = t+MU*(1.0-t);   /* new cutting plane */
        /*        t2 = t+(1.0-t)/10.0;   */

        /* update W to be the best so far solution */
        sq_norm_W = update_W( t1, user_data );

        /* select a new cut */
        xi = 0;
        cut_length = 0;
        ocas.trn_err = 0;
        for(i=0; i < nData; i++ ) {

          if( (old_output[i]*(1-t2) + t2*output[i]) <= 1 )
          {
            new_cut[cut_length] = i;
            cut_length++;
          }

          output[i] = old_output[i]*(1-t1) + t1*output[i];

          if( output[i] <= 1) xi += 1-output[i];
          if( output[i] <= 0) ocas.trn_err++;

        }

        ocas.Q_P = 0.5*sq_norm_W + C*xi;

        ocas.ocas_time = get_time() - ocas_start_time;

        /*        ocas_print("%4d: tim=%f, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, 1-Q_D/Q_P=%f, nza=%4d, err=%.2f%%, qpf=%d\n",
                   ocas.nIter, cur_time, ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/LIBOCAS_ABS(ocas.Q_P),
                   ocas.nNZAlpha, 100*(float64_t)ocas.trn_err/(float64_t)nData, ocas.qp_exitflag );
        */

        start_time = get_time();
        ocas_print(ocas);
        ocas.print_time += get_time() - start_time;

        break;
    }

    /* Stopping conditions */
    if( ocas.Q_P - ocas.Q_D <= TolRel*LIBOCAS_ABS(ocas.Q_P)) ocas.exitflag = 1;
    if( ocas.Q_P - ocas.Q_D <= TolAbs) ocas.exitflag = 2;
    if( ocas.Q_P <= QPBound) ocas.exitflag = 3;
    if( MaxTime > 0 && ocas.ocas_time >= MaxTime) ocas.exitflag = 4;
    if(ocas.nCutPlanes >= BufSize) ocas.exitflag = -1;

  } /* end of the main loop */

cleanup:

  LIBOCAS_FREE(H);
  LIBOCAS_FREE(b);
  LIBOCAS_FREE(alpha);
  LIBOCAS_FREE(new_cut);
  LIBOCAS_FREE(I);
  LIBOCAS_FREE(diag_H);
  LIBOCAS_FREE(output);
  LIBOCAS_FREE(old_output);
  LIBOCAS_FREE(hpf);
/*  LIBOCAS_FREE(hpi);*/
  LIBOCAS_FREE(hpb);
  LIBOCAS_FREE(Ci);
  LIBOCAS_FREE(Bi);

  ocas.ocas_time = get_time() - ocas_start_time;

  return(ocas);
}


/*----------------------------------------------------------------------
  Binary linear Ocas-SVM solver which allows using different C for each
  training example.
  ----------------------------------------------------------------------*/
ocas_return_value_T svm_ocas_solver_difC(
            float64_t *C,
            uint32_t nData,
            float64_t TolRel,
            float64_t TolAbs,
            float64_t QPBound,
            float64_t MaxTime,
            uint32_t _BufSize,
            uint8_t Method,
            void (*compute_W)(float64_t*, float64_t*, float64_t*, uint32_t, void*),
            float64_t (*update_W)(float64_t, void*),
            int (*add_new_cut)(float64_t*, uint32_t*, uint32_t, uint32_t, void*),
            int (*compute_output)(float64_t*, void* ),
            int (*sort)(float64_t*, float64_t*, uint32_t),
			void (*ocas_print)(ocas_return_value_T),
			void* user_data)
{
  ocas_return_value_T ocas={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  float64_t *b, *alpha, *diag_H;
  float64_t *output, *old_output;
  float64_t xi, sq_norm_W, QPSolverTolRel, dot_prod_WoldW, sq_norm_oldW;
  float64_t A0, B0, GradVal, t, t1, t2, *Ci, *Bi, *hpf, *hpb;
  float64_t start_time, ocas_start_time;
  float64_t qp_b = 1.0;
  float64_t new_b;
  uint32_t cut_length;
  uint32_t i, *new_cut;
  uint32_t *I;
  uint8_t S = 1;
  libqp_state_T qp_exitflag;

  ocas_start_time = get_time();
  ocas.qp_solver_time = 0;
  ocas.output_time = 0;
  ocas.sort_time = 0;
  ocas.add_time = 0;
  ocas.w_time = 0;
  ocas.print_time = 0;

  BufSize = _BufSize;

  QPSolverTolRel = TolRel*0.5;

  H=NULL;
  b=NULL;
  alpha=NULL;
  new_cut=NULL;
  I=NULL;
  diag_H=NULL;
  output=NULL;
  old_output=NULL;
  hpf=NULL;
  hpb = NULL;
  Ci=NULL;
  Bi=NULL;

  /* Hessian matrix contains dot product of normal vectors of selected cutting planes */
  H = (float64_t*)LIBOCAS_CALLOC(BufSize*BufSize, float64_t);
  if(H == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* bias of cutting planes */
  b = (float64_t*)LIBOCAS_CALLOC(BufSize, float64_t);
  if(b == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  alpha = (float64_t*)LIBOCAS_CALLOC(BufSize, float64_t);
  if(alpha == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* indices of examples which define a new cut */
  new_cut = (uint32_t*)LIBOCAS_CALLOC(nData, uint32_t);
  if(new_cut == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  I = (uint32_t*)LIBOCAS_CALLOC(BufSize, uint32_t);
  if(I == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  for(i=0; i< BufSize; i++) I[i] = 1;

  diag_H = (float64_t*)LIBOCAS_CALLOC(BufSize, float64_t);
  if(diag_H == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  output = (float64_t*)LIBOCAS_CALLOC(nData, float64_t);
  if(output == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  old_output = (float64_t*)LIBOCAS_CALLOC(nData, float64_t);
  if(old_output == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* array of hinge points used in line-serach  */
  hpf = (float64_t*) LIBOCAS_CALLOC(nData, float64_t);
  if(hpf == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  hpb = (float64_t*) LIBOCAS_CALLOC(nData, float64_t);
  if(hpb == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* vectors Ci, Bi are used in the line search procedure */
  Ci = (float64_t*)LIBOCAS_CALLOC(nData, float64_t);
  if(Ci == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  Bi = (float64_t*)LIBOCAS_CALLOC(nData, float64_t);
  if(Bi == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  ocas.nCutPlanes = 0;
  ocas.exitflag = 0;
  ocas.nIter = 0;

  /* Compute initial value of Q_P assuming that W is zero vector.*/
  sq_norm_W = 0;
/*
  xi = nData;
  ocas.Q_P = 0.5*sq_norm_W + C*xi;
*/
  ocas.Q_D = 0;

  /* Compute the initial cutting plane */
  cut_length = nData;
  new_b = 0;
  for(i=0; i < nData; i++)
  {
    new_cut[i] = i;
    new_b += C[i];
  }

  ocas.Q_P = 0.5*sq_norm_W + new_b;


  ocas.trn_err = nData;
  ocas.ocas_time = get_time() - ocas_start_time;
  /*  ocas_print("%4d: tim=%f, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, Q_P-Q_D/abs(Q_P)=%f\n",
          ocas.nIter,cur_time, ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/LIBOCAS_ABS(ocas.Q_P));
  */
  ocas_print(ocas);

  /* main loop */
  while( ocas.exitflag == 0 )
  {
    ocas.nIter++;

    /* append a new cut to the buffer and update H */
/*    b[ocas.nCutPlanes] = -(float64_t)cut_length*C;*/
    b[ocas.nCutPlanes] = -new_b;

    start_time = get_time();

    if(add_new_cut( &H[LIBOCAS_INDEX(0,ocas.nCutPlanes,BufSize)], new_cut, cut_length, ocas.nCutPlanes, user_data ) != 0)
    {
	  ocas.exitflag=-2;
	  goto cleanup;
    }

    ocas.add_time += get_time() - start_time;

    /* copy new added row:  H(ocas.nCutPlanes,ocas.nCutPlanes,1:ocas.nCutPlanes-1) = H(1:ocas.nCutPlanes-1:ocas.nCutPlanes)' */
    diag_H[ocas.nCutPlanes] = H[LIBOCAS_INDEX(ocas.nCutPlanes,ocas.nCutPlanes,BufSize)];
    for(i=0; i < ocas.nCutPlanes; i++) {
      H[LIBOCAS_INDEX(ocas.nCutPlanes,i,BufSize)] = H[LIBOCAS_INDEX(i,ocas.nCutPlanes,BufSize)];
    }

    ocas.nCutPlanes++;

    /* call inner QP solver */
    start_time = get_time();

/*    qp_exitflag = libqp_splx_solver(&get_col, diag_H, b, &C, I, &S, alpha,*/
/*                                  ocas.nCutPlanes, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBOCAS_PLUS_INF,0);*/
    qp_exitflag = libqp_splx_solver(&get_col, diag_H, b, &qp_b, I, &S, alpha,
                                  ocas.nCutPlanes, QPSolverMaxIter, 0.0, QPSolverTolRel, -LIBOCAS_PLUS_INF,0);

    ocas.qp_exitflag = qp_exitflag.exitflag;

    ocas.qp_solver_time += get_time() - start_time;
    ocas.Q_D = -qp_exitflag.QP;

    ocas.nNZAlpha = 0;
    for(i=0; i < ocas.nCutPlanes; i++) {
      if( alpha[i] != 0) ocas.nNZAlpha++;
    }

    sq_norm_oldW = sq_norm_W;
    start_time = get_time();
    compute_W( &sq_norm_W, &dot_prod_WoldW, alpha, ocas.nCutPlanes, user_data );
    ocas.w_time += get_time() - start_time;

    /* select a new cut */
    switch( Method )
    {
      /* cutting plane algorithm implemented in SVMperf and BMRM */
      case 0:

        start_time = get_time();
        if( compute_output( output, user_data ) != 0)
        {
          ocas.exitflag=-2;
          goto cleanup;
        }
        ocas.output_time += get_time()-start_time;

        xi = 0;
        cut_length = 0;
        ocas.trn_err = 0;
        new_b = 0;
        for(i=0; i < nData; i++)
        {
          if(output[i] <= 0) ocas.trn_err++;

/*          if(output[i] <= 1) {*/
/*            xi += 1 - output[i];*/
          if(output[i] <= C[i]) {
            xi += C[i] - output[i];
            new_cut[cut_length] = i;
            cut_length++;
            new_b += C[i];
          }
        }
/*        ocas.Q_P = 0.5*sq_norm_W + C*xi;*/
        ocas.Q_P = 0.5*sq_norm_W + xi;

        ocas.ocas_time = get_time() - ocas_start_time;

        /*        ocas_print("%4d: tim=%f, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, 1-Q_D/Q_P=%f, nza=%4d, err=%.2f%%, qpf=%d\n",
                  ocas.nIter,cur_time, ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/LIBOCAS_ABS(ocas.Q_P),
                  ocas.nNZAlpha, 100*(float64_t)ocas.trn_err/(float64_t)nData, ocas.qp_exitflag );
        */

        start_time = get_time();
        ocas_print(ocas);
        ocas.print_time += get_time() - start_time;

        break;


      /* Ocas strategy */
      case 1:

        /* Linesearch */
        A0 = sq_norm_W -2*dot_prod_WoldW + sq_norm_oldW;
        B0 = dot_prod_WoldW - sq_norm_oldW;

        memcpy( old_output, output, sizeof(float64_t)*nData );

        start_time = get_time();
        if( compute_output( output, user_data ) != 0)
        {
          ocas.exitflag=-2;
          goto cleanup;
        }
        ocas.output_time += get_time()-start_time;

        uint32_t num_hp = 0;
        GradVal = B0;
        for(i=0; i< nData; i++) {

/*          Ci[i] = C*(1-old_output[i]);*/
/*          Bi[i] = C*(old_output[i] - output[i]);*/
          Ci[i] = (C[i]-old_output[i]);
          Bi[i] = old_output[i] - output[i];

          float64_t val;
          if(Bi[i] != 0)
            val = -Ci[i]/Bi[i];
          else
            val = -LIBOCAS_PLUS_INF;

          if (val>0)
          {
/*            hpi[num_hp] = i;*/
            hpb[num_hp] = Bi[i];
            hpf[num_hp] = val;
            num_hp++;
          }

          if( (Bi[i] < 0 && val > 0) || (Bi[i] > 0 && val <= 0))
            GradVal += Bi[i];

        }

        t = 0;
        if( GradVal < 0 )
        {
          start_time = get_time();
/*          if( sort(hpf, hpi, num_hp) != 0)*/
          if( sort(hpf, hpb, num_hp) != 0 )
          {
            ocas.exitflag=-2;
            goto cleanup;
          }
          ocas.sort_time += get_time() - start_time;

          float64_t t_new, GradVal_new;
          i = 0;
          while( GradVal < 0 && i < num_hp )
          {
            t_new = hpf[i];
            GradVal_new = GradVal + LIBOCAS_ABS(hpb[i]) + A0*(t_new-t);

            if( GradVal_new >= 0 )
            {
              t = t + GradVal*(t-t_new)/(GradVal_new - GradVal);
            }
            else
            {
              t = t_new;
              i++;
            }

            GradVal = GradVal_new;
          }
        }

        /*
        t = hpf[0] - 1;
        i = 0;
        GradVal = t*A0 + Bsum;
        while( GradVal < 0 && i < num_hp && hpf[i] < LIBOCAS_PLUS_INF ) {
          t = hpf[i];
          Bsum = Bsum + LIBOCAS_ABS(Bi[hpi[i]]);
          GradVal = t*A0 + Bsum;
          i++;
        }
        */
        t = LIBOCAS_MAX(t,0);          /* just sanity check; t < 0 should not ocure */

        t1 = t;                /* new (best so far) W */
        t2 = t+(1.0-t)*MU;   /* new cutting plane */
        /*        t2 = t+(1.0-t)/10.0;   new cutting plane */

        /* update W to be the best so far solution */
        sq_norm_W = update_W( t1, user_data );

        /* select a new cut */
        xi = 0;
        cut_length = 0;
        ocas.trn_err = 0;
        new_b = 0;
        for(i=0; i < nData; i++ ) {

/*          if( (old_output[i]*(1-t2) + t2*output[i]) <= 1 ) */
          if( (old_output[i]*(1-t2) + t2*output[i]) <= C[i] )
          {
            new_cut[cut_length] = i;
            cut_length++;
            new_b += C[i];
          }

          output[i] = old_output[i]*(1-t1) + t1*output[i];

/*          if( output[i] <= 1) xi += 1-output[i];*/
          if( output[i] <= C[i]) xi += C[i]-output[i];
          if( output[i] <= 0) ocas.trn_err++;

        }

/*        ocas.Q_P = 0.5*sq_norm_W + C*xi;*/
        ocas.Q_P = 0.5*sq_norm_W + xi;

        ocas.ocas_time = get_time() - ocas_start_time;

        /*        ocas_print("%4d: tim=%f, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, 1-Q_D/Q_P=%f, nza=%4d, err=%.2f%%, qpf=%d\n",
                   ocas.nIter, cur_time, ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/LIBOCAS_ABS(ocas.Q_P),
                   ocas.nNZAlpha, 100*(float64_t)ocas.trn_err/(float64_t)nData, ocas.qp_exitflag );
        */

        start_time = get_time();
        ocas_print(ocas);
        ocas.print_time += get_time() - start_time;

        break;
    }

    /* Stopping conditions */
    if( ocas.Q_P - ocas.Q_D <= TolRel*LIBOCAS_ABS(ocas.Q_P)) ocas.exitflag = 1;
    if( ocas.Q_P - ocas.Q_D <= TolAbs) ocas.exitflag = 2;
    if( ocas.Q_P <= QPBound) ocas.exitflag = 3;
    if( MaxTime > 0 && ocas.ocas_time >= MaxTime) ocas.exitflag = 4;
    if(ocas.nCutPlanes >= BufSize) ocas.exitflag = -1;

  } /* end of the main loop */

cleanup:

  LIBOCAS_FREE(H);
  LIBOCAS_FREE(b);
  LIBOCAS_FREE(alpha);
  LIBOCAS_FREE(new_cut);
  LIBOCAS_FREE(I);
  LIBOCAS_FREE(diag_H);
  LIBOCAS_FREE(output);
  LIBOCAS_FREE(old_output);
  LIBOCAS_FREE(hpf);
/*  LIBOCAS_FREE(hpi);*/
  LIBOCAS_FREE(hpb);
  LIBOCAS_FREE(Ci);
  LIBOCAS_FREE(Bi);

  ocas.ocas_time = get_time() - ocas_start_time;

  return(ocas);
}



/*----------------------------------------------------------------------
  Multiclass SVM-Ocas solver
  ----------------------------------------------------------------------*/

/* Helper function needed by the multi-class SVM linesearch.

  - This function finds a simplified representation of a piece-wise linear function
  by splitting the domain into intervals and fining active terms for these intevals */
static void findactive(float64_t *Theta, float64_t *SortedA, uint32_t *nSortedA, float64_t *A, float64_t *B, int n,
            int (*sort)(float64_t*, float64_t*, uint32_t))
{
  float64_t tmp, theta;
  uint32_t i, j, idx, idx2 = 0, start;

  sort(A,B,n);

  idx = 0;
  i = 0;
  while( i < (uint32_t)n-1 && A[i] == A[i+1])
  {
    if( B[i+1] > B[idx] )
    {
      idx = i+1;
    }
    i++;
  }

  (*nSortedA) = 1;
  SortedA[0] = A[idx];

  while(1)
  {
    start = idx + 1;
    while( start < (uint32_t)n && A[idx] == A[start])
      start++;

    theta = LIBOCAS_PLUS_INF;
    for(j=start; j < (uint32_t)n; j++)
    {
      tmp = (B[j] - B[idx])/(A[idx]-A[j]);
      if( tmp < theta)
      {
        theta = tmp;
        idx2 = j;
      }
    }

    if( theta < LIBOCAS_PLUS_INF)
    {
      Theta[(*nSortedA) - 1] = theta;
      SortedA[(*nSortedA)] = A[idx2];
      (*nSortedA)++;
      idx = idx2;
    }
    else
      return;
  }
}


/*----------------------------------------------------------------------
  Multiclass linear OCAS-SVM solver.
  ----------------------------------------------------------------------*/
ocas_return_value_T msvm_ocas_solver(
            float64_t C,
            float64_t *data_y,
            uint32_t nY,
            uint32_t nData,
            float64_t TolRel,
            float64_t TolAbs,
            float64_t QPBound,
            float64_t MaxTime,
            uint32_t _BufSize,
            uint8_t Method,
            void (*compute_W)(float64_t*, float64_t*, float64_t*, uint32_t, void*),
            float64_t (*update_W)(float64_t, void*),
            int (*add_new_cut)(float64_t*, uint32_t*, uint32_t, void*),
            int (*compute_output)(float64_t*, void* ),
            int (*sort)(float64_t*, float64_t*, uint32_t),
			void (*ocas_print)(ocas_return_value_T),
			void* user_data)
{
  ocas_return_value_T ocas={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  float64_t *b, *alpha, *diag_H;
  float64_t *output, *old_output;
  float64_t xi, sq_norm_W, QPSolverTolRel, QPSolverTolAbs, dot_prod_WoldW, sq_norm_oldW;
  float64_t A0, B0, t, t1, t2, R, tmp, element_b, x;
  float64_t *A, *B, *theta, *Theta, *sortedA, *Add;
  float64_t start_time, ocas_start_time, grad_sum, grad, min_x = 0, old_x, old_grad;
  uint32_t i, y, y2, ypred = 0, *new_cut, cnt1, cnt2, j, nSortedA, idx;
  uint32_t *I;
  uint8_t S = 1;
  libqp_state_T qp_exitflag;

  ocas_start_time = get_time();
  ocas.qp_solver_time = 0;
  ocas.output_time = 0;
  ocas.sort_time = 0;
  ocas.add_time = 0;
  ocas.w_time = 0;
  ocas.print_time = 0;

  BufSize = _BufSize;

  QPSolverTolRel = TolRel*0.5;
  QPSolverTolAbs = TolAbs*0.5;

  H=NULL;
  b=NULL;
  alpha=NULL;
  new_cut=NULL;
  I=NULL;
  diag_H=NULL;
  output=NULL;
  old_output=NULL;
  A = NULL;
  B = NULL;
  theta = NULL;
  Theta = NULL;
  sortedA = NULL;
  Add = NULL;

  /* Hessian matrix contains dot product of normal vectors of selected cutting planes */
  H = (float64_t*)LIBOCAS_CALLOC(BufSize*BufSize, float64_t);
  if(H == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* bias of cutting planes */
  b = (float64_t*)LIBOCAS_CALLOC(BufSize, float64_t);
  if(b == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  alpha = (float64_t*)LIBOCAS_CALLOC(BufSize, float64_t);
  if(alpha == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* indices of examples which define a new cut */
  new_cut = (uint32_t*)LIBOCAS_CALLOC(nData, uint32_t);
  if(new_cut == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  I = (uint32_t*)LIBOCAS_CALLOC(BufSize, uint32_t);
  if(I == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  for(i=0; i< BufSize; i++)
    I[i] = 1;

  diag_H = (float64_t*)LIBOCAS_CALLOC(BufSize, float64_t);
  if(diag_H == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  output = (float64_t*)LIBOCAS_CALLOC(nData*nY, float64_t);
  if(output == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  old_output = (float64_t*)LIBOCAS_CALLOC(nData*nY, float64_t);
  if(old_output == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* auxciliary variables used in the linesearch */
  A = (float64_t*)LIBOCAS_CALLOC(nData*nY, float64_t);
  if(A == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  B = (float64_t*)LIBOCAS_CALLOC(nData*nY, float64_t);
  if(B == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  theta = (float64_t*)LIBOCAS_CALLOC(nY, float64_t);
  if(theta == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  sortedA = (float64_t*)LIBOCAS_CALLOC(nY, float64_t);
  if(sortedA == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  Theta = (float64_t*)LIBOCAS_CALLOC(nData*nY, float64_t);
  if(Theta == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  Add = (float64_t*)LIBOCAS_CALLOC(nData*nY, float64_t);
  if(Add == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* Set initial values*/
  ocas.nCutPlanes = 0;
  ocas.exitflag = 0;
  ocas.nIter = 0;
  ocas.Q_D = 0;
  ocas.trn_err = nData;
  R = (float64_t)nData;
  sq_norm_W = 0;
  element_b = (float64_t)nData;
  ocas.Q_P = 0.5*sq_norm_W + C*R;

  /* initial cutting plane */
  for(i=0; i < nData; i++)
  {
    y2 = (uint32_t)data_y[i];

    if(y2 > 0)
      new_cut[i] = 0;
    else
      new_cut[i] = 1;

  }

  ocas.ocas_time = get_time() - ocas_start_time;

  start_time = get_time();
  ocas_print(ocas);
  ocas.print_time += get_time() - start_time;

  /* main loop of the OCAS */
  while( ocas.exitflag == 0 )
  {
    ocas.nIter++;

    /* append a new cut to the buffer and update H */
    b[ocas.nCutPlanes] = -(float64_t)element_b;

    start_time = get_time();

    if(add_new_cut( &H[LIBOCAS_INDEX(0,ocas.nCutPlanes,BufSize)], new_cut, ocas.nCutPlanes, user_data ) != 0)
    {
	  ocas.exitflag=-2;
	  goto cleanup;
    }

    ocas.add_time += get_time() - start_time;

    /* copy newly appended row: H(ocas.nCutPlanes,ocas.nCutPlanes,1:ocas.nCutPlanes-1) = H(1:ocas.nCutPlanes-1:ocas.nCutPlanes)' */
    diag_H[ocas.nCutPlanes] = H[LIBOCAS_INDEX(ocas.nCutPlanes,ocas.nCutPlanes,BufSize)];
    for(i=0; i < ocas.nCutPlanes; i++)
    {
      H[LIBOCAS_INDEX(ocas.nCutPlanes,i,BufSize)] = H[LIBOCAS_INDEX(i,ocas.nCutPlanes,BufSize)];
    }

    ocas.nCutPlanes++;

    /* call inner QP solver */
    start_time = get_time();

    qp_exitflag = libqp_splx_solver(&get_col, diag_H, b, &C, I, &S, alpha,
                                  ocas.nCutPlanes, QPSolverMaxIter, QPSolverTolAbs, QPSolverTolRel, -LIBOCAS_PLUS_INF,0);

    ocas.qp_exitflag = qp_exitflag.exitflag;

    ocas.qp_solver_time += get_time() - start_time;
    ocas.Q_D = -qp_exitflag.QP;

    ocas.nNZAlpha = 0;
    for(i=0; i < ocas.nCutPlanes; i++)
      if( alpha[i] != 0) ocas.nNZAlpha++;

    sq_norm_oldW = sq_norm_W;
    start_time = get_time();
    compute_W( &sq_norm_W, &dot_prod_WoldW, alpha, ocas.nCutPlanes, user_data );
    ocas.w_time += get_time() - start_time;

    /* select a new cut */
    switch( Method )
    {
      /* cutting plane algorithm implemented in SVMperf and BMRM */
      case 0:

        start_time = get_time();
        if( compute_output( output, user_data ) != 0)
        {
          ocas.exitflag=-2;
          goto cleanup;
        }
        ocas.output_time += get_time()-start_time;

        /* the following loop computes: */
        element_b = 0.0;    /*  element_b = R(old_W) - g'*old_W */
        R = 0;              /*  R(W) = sum_i max_y ( [[y != y_i]] + (w_y- w_y_i)'*x_i )    */
        ocas.trn_err = 0;   /*  trn_err = sum_i [[y != y_i ]]                              */
                            /* new_cut[i] = argmax_i ( [[y != y_i]] + (w_y- w_y_i)'*x_i )  */
        for(i=0; i < nData; i++)
        {
          y2 = (uint32_t)data_y[i];

          for(xi=-LIBOCAS_PLUS_INF, y=0; y < nY; y++)
          {
            if(y2 != y && xi < output[LIBOCAS_INDEX(y,i,nY)])
            {
              xi = output[LIBOCAS_INDEX(y,i,nY)];
              ypred = y;
            }
          }

          if(xi >= output[LIBOCAS_INDEX(y2,i,nY)])
            ocas.trn_err ++;

          xi = LIBOCAS_MAX(0,xi+1-output[LIBOCAS_INDEX(y2,i,nY)]);
          R += xi;
          if(xi > 0)
          {
            element_b++;
            new_cut[i] = ypred;
          }
          else
            new_cut[i] = y2;
        }

        ocas.Q_P = 0.5*sq_norm_W + C*R;

        ocas.ocas_time = get_time() - ocas_start_time;

        start_time = get_time();
        ocas_print(ocas);
        ocas.print_time += get_time() - start_time;

        break;

      /* The OCAS solver */
      case 1:
        memcpy( old_output, output, sizeof(float64_t)*nData*nY );

        start_time = get_time();
        if( compute_output( output, user_data ) != 0)
        {
          ocas.exitflag=-2;
          goto cleanup;
        }
        ocas.output_time += get_time()-start_time;

        A0 = sq_norm_W - 2*dot_prod_WoldW + sq_norm_oldW;
        B0 = dot_prod_WoldW - sq_norm_oldW;

        for(i=0; i < nData; i++)
        {
          y2 = (uint32_t)data_y[i];

          for(y=0; y < nY; y++)
          {
            A[LIBOCAS_INDEX(y,i,nY)] = C*(output[LIBOCAS_INDEX(y,i,nY)] - old_output[LIBOCAS_INDEX(y,i,nY)]
                                       + old_output[LIBOCAS_INDEX(y2,i,nY)] - output[LIBOCAS_INDEX(y2,i,nY)]);
            B[LIBOCAS_INDEX(y,i,nY)] = C*(old_output[LIBOCAS_INDEX(y,i,nY)] - old_output[LIBOCAS_INDEX(y2,i,nY)]
                                       + (float64_t)(y != y2));
          }
        }

        /* linesearch */
/*      new_x = msvm_linesearch_mex(A0,B0,AA*C,BB*C);*/

        grad_sum = B0;
        cnt1 = 0;
        cnt2 = 0;
        for(i=0; i < nData; i++)
        {
          findactive(theta,sortedA,&nSortedA,&A[i*nY],&B[i*nY],nY,sort);

          idx = 0;
          while( idx < nSortedA-1 && theta[idx] < 0 )
            idx++;

          grad_sum += sortedA[idx];

          for(j=idx; j < nSortedA-1; j++)
          {
            Theta[cnt1] = theta[j];
            cnt1++;
          }

          for(j=idx+1; j < nSortedA; j++)
          {
            Add[cnt2] = -sortedA[j-1]+sortedA[j];
            cnt2++;
          }
        }

        start_time = get_time();
        sort(Theta,Add,cnt1);
        ocas.sort_time += get_time() - start_time;

        grad = grad_sum;
        if(grad >= 0)
        {
          min_x = 0;
        }
        else
        {
          old_x = 0;
          old_grad = grad;

          for(i=0; i < cnt1; i++)
          {
            x = Theta[i];

            grad = x*A0 + grad_sum;

            if(grad >=0)
            {

              min_x = (grad*old_x - old_grad*x)/(grad - old_grad);

              break;
            }
            else
            {
              grad_sum = grad_sum + Add[i];

              grad = x*A0 + grad_sum;
              if( grad >= 0)
              {
                min_x = x;
                break;
              }
            }

            old_grad = grad;
            old_x = x;
          }
        }
        /* end of the linesearch which outputs min_x */

        t = min_x;
        t1 = t;                /* new (best so far) W */
        t2 = t+(1.0-t)*MU;   /* new cutting plane */
        /*        t2 = t+(1.0-t)/10.0;    */

        /* update W to be the best so far solution */
        sq_norm_W = update_W( t1, user_data );

        /* the following code  computes a new cutting plane: */
        element_b = 0.0;    /*  element_b = R(old_W) - g'*old_W */
                            /* new_cut[i] = argmax_i ( [[y != y_i]] + (w_y- w_y_i)'*x_i )  */
        for(i=0; i < nData; i++)
        {
          y2 = (uint32_t)data_y[i];

          for(xi=-LIBOCAS_PLUS_INF, y=0; y < nY; y++)
          {
            tmp = old_output[LIBOCAS_INDEX(y,i,nY)]*(1-t2) + t2*output[LIBOCAS_INDEX(y,i,nY)];
            if(y2 != y && xi < tmp)
            {
              xi = tmp;
              ypred = y;
            }
          }

          tmp = old_output[LIBOCAS_INDEX(y2,i,nY)]*(1-t2) + t2*output[LIBOCAS_INDEX(y2,i,nY)];
          xi = LIBOCAS_MAX(0,xi+1-tmp);
          if(xi > 0)
          {
            element_b++;
            new_cut[i] = ypred;
          }
          else
            new_cut[i] = y2;
        }

        /* compute Risk, class. error and update outputs to correspond to the new W */
        ocas.trn_err = 0;   /*  trn_err = sum_i [[y != y_i ]]                       */
        R = 0;
        for(i=0; i < nData; i++)
        {
          y2 = (uint32_t)data_y[i];

          for(tmp=-LIBOCAS_PLUS_INF, y=0; y < nY; y++)
          {
            output[LIBOCAS_INDEX(y,i,nY)] = old_output[LIBOCAS_INDEX(y,i,nY)]*(1-t1) + t1*output[LIBOCAS_INDEX(y,i,nY)];

            if(y2 != y && tmp < output[LIBOCAS_INDEX(y,i,nY)])
            {
              ypred = y;
              tmp = output[LIBOCAS_INDEX(y,i,nY)];
            }
          }

          R += LIBOCAS_MAX(0,1+tmp - output[LIBOCAS_INDEX(y2,i,nY)]);
          if( tmp >= output[LIBOCAS_INDEX(y2,i,nY)])
            ocas.trn_err ++;
        }

        ocas.Q_P = 0.5*sq_norm_W + C*R;


        /* get time and print status */
        ocas.ocas_time = get_time() - ocas_start_time;

        start_time = get_time();
        ocas_print(ocas);
        ocas.print_time += get_time() - start_time;

        break;

    }

    /* Stopping conditions */
    if( ocas.Q_P - ocas.Q_D <= TolRel*LIBOCAS_ABS(ocas.Q_P)) ocas.exitflag = 1;
    if( ocas.Q_P - ocas.Q_D <= TolAbs) ocas.exitflag = 2;
    if( ocas.Q_P <= QPBound) ocas.exitflag = 3;
    if( MaxTime > 0 && ocas.ocas_time >= MaxTime) ocas.exitflag = 4;
    if(ocas.nCutPlanes >= BufSize) ocas.exitflag = -1;

  } /* end of the main loop */

cleanup:

  LIBOCAS_FREE(H);
  LIBOCAS_FREE(b);
  LIBOCAS_FREE(alpha);
  LIBOCAS_FREE(new_cut);
  LIBOCAS_FREE(I);
  LIBOCAS_FREE(diag_H);
  LIBOCAS_FREE(output);
  LIBOCAS_FREE(old_output);
  LIBOCAS_FREE(A);
  LIBOCAS_FREE(B);
  LIBOCAS_FREE(theta);
  LIBOCAS_FREE(Theta);
  LIBOCAS_FREE(sortedA);
  LIBOCAS_FREE(Add);

  ocas.ocas_time = get_time() - ocas_start_time;

  return(ocas);
}
}


