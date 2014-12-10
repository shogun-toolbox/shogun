/*-----------------------------------------------------------------------
 * libqp_splx.c: solver for Quadratic Programming task with 
 * simplex constraints.
 *
 * DESCRIPTION
 *  The library provides function which solves the following instance of
 *  a convex Quadratic Programmin task:
 *  
 *   min QP(x):= 0.5*x'*H*x + f'*x  
 *    x
 *
 * subject to:   
 *   sum_{i in I_k} x[i] == b[k]  for all k such that S[k] == 0 
 *   sum_{i in I_k} x[i] <= b[k]  for all k such that S[k] == 1
 *                             x(i) >= 0 for all i=1:n
 *   
 *  where I_k = { i | I[i] == k}, k={1,...,m}.
 *
 * A precision of the found solution is controled by the input argumens
 * MaxIter, TolAbs, QP_TH and MaxIter which define the stopping conditions:
 * 
 *  nIter >= MaxIter     ->  exitflag = 0   Number of iterations
 *  QP-QD <= TolAbs      ->  exitflag = 1   Abs. tolerance (duality gap)
 *  QP-QD <= QP*TolRel   ->  exitflag = 2   Relative tolerance
 *  QP <= QP_TH          ->  exitflag = 3   Threshold on objective value
 *
 * where QP and QD are primal respectively dual objective values.
 *
 * INPUT ARGUMENTS
 *  get_col   function which returns pointer to the i-th column of H.
 *  diag_H [float64_t n x 1] vector containing values on the diagonal of H.
 *  f [float64_t n x 1] vector.
 *  b [float64_t n x 1] vector of positive numbers.
 *  I [uint16_T n x 1] vector containing numbers 1...m. 
 *  S [uint8_T n x 1] vector containing numbers 0 and 1.
 *  x [float64_t n x 1] solution vector; must be feasible.
 *  n [uint32_t 1 x 1] dimension of H.
 *  MaxIter [uint32_t 1 x 1] max number of iterations.
 *  TolAbs [float64_t 1 x 1] Absolute tolerance.
 *  TolRel [float64_t 1 x 1] Relative tolerance.
 *  QP_TH  [float64_t 1 x 1] Threshold on the primal value.
 *  print_state  print function; if == NULL it is not called.
 *
 * RETURN VALUE
 *  structure [libqp_state_T] 
 *  .QP [1 x 1] Primal objective value.
 *  .QD [1 x 1] Dual objective value.
 *  .nIter [1 x 1] Number of iterations.
 *  .exitflag [1 x 1] Indicates which stopping condition was used:
 *    -1  ... Not enough memory.
 *     0  ... Maximal number of iteations reached: nIter >= MaxIter.
 *     1  ... Relarive tolerance reached: QP-QD <= abs(QP)*TolRel
 *     2  ... Absolute tolerance reached: QP-QD <= TolAbs
 *     3  ... Objective value reached threshold: QP <= QP_TH.
 *
 * REFERENCE
 *  The algorithm is described in:
 *  V. Franc, V. Hlavac. A Novel Algorithm for Learning Support Vector Machines
 *   with Structured Output Spaces. Research Report K333 22/06, CTU-CMP-2006-04. 
 *   May, 2006. ftp://cmp.felk.cvut.cz/pub/cmp/articles/franc/Franc-TR-2006-04.ps
 *
 * Copyright (C) 2006-2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Center for Machine Perception, CTU FEL Prague
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public 
 * License as published by the Free Software Foundation; 
 * Version 3, 29 June 2007
 *-------------------------------------------------------------------- */

#include <shogun/mathematics/Math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#include <shogun/lib/common.h>
#include <shogun/lib/external/libqp.h>
namespace shogun
{

libqp_state_T libqp_splx_solver(const float64_t* (*get_col)(uint32_t),
                  float64_t *diag_H,
                  float64_t *f,
                  float64_t *b,
                  uint32_t *I,
                  uint8_t *S,
                  float64_t *x,
                  uint32_t n,
                  uint32_t MaxIter,
                  float64_t TolAbs,
                  float64_t TolRel,
                  float64_t QP_TH,
				  void (*print_state)(libqp_state_T state))
{
  float64_t *d;
  float64_t *col_u, *col_v;
  float64_t *x_neq;
  float64_t tmp;
  float64_t improv;
  float64_t tmp_num;
  float64_t tmp_den=0;
  float64_t tau=0;
  float64_t delta;
  uint32_t *inx;
  uint32_t *nk;
  uint32_t m;
  int32_t u=0;
  int32_t v=0;
  uint32_t k;
  uint32_t i, j;
  libqp_state_T state;

  
  /* ------------------------------------------------------------ 
    Initialization                                               
  ------------------------------------------------------------ */
  state.nIter = 0;
  state.QP = LIBQP_PLUS_INF;
  state.QD = -LIBQP_PLUS_INF;
  state.exitflag = 100;

  inx=NULL;
  nk=NULL;
  d=NULL;
  x_neq = NULL;

  /* count number of constraints */
  for( i=0, m=0; i < n; i++ ) 
    m = LIBQP_MAX(m,I[i]);

  /* auxciliary variables for tranforming equalities to inequalities */
  x_neq = (float64_t*) LIBQP_CALLOC(m, float64_t);
  if( x_neq == NULL )
  {
	  state.exitflag=-1;
	  goto cleanup;
  }

  /* inx is translation table between variable index i and its contraint */
  inx = (uint32_t*) LIBQP_CALLOC(m*n, uint32_t);
  if( inx == NULL )
  {
	  state.exitflag=-1;
	  goto cleanup;
  }

  /* nk is the number of variables coupled by i-th linear constraint */
  nk = (uint32_t*) LIBQP_CALLOC(m, uint32_t);
  if( nk == NULL )
  {
	  state.exitflag=-1;
	  goto cleanup;
  }

  /* setup auxciliary variables */
  for( i=0; i < m; i++ ) 
    x_neq[i] = b[i];


  /* create inx and nk */
  for( i=0; i < n; i++ ) {
     k = I[i]-1;
     inx[LIBQP_INDEX(nk[k],k,n)] = i;
     nk[k]++;     

     if(S[k] != 0) 
       x_neq[k] -= x[i];
  }
    
  /* d = H*x + f is gradient*/
  d = (float64_t*) LIBQP_CALLOC(n, float64_t);
  if( d == NULL )
  {
	  state.exitflag=-1;
	  goto cleanup;
  }
 
  /* compute gradient */
  for( i=0; i < n; i++ ) 
  {
    d[i] += f[i];
    if( x[i] > 0 ) {
      col_u = (float64_t*)get_col(i);      
      for( j=0; j < n; j++ ) {
          d[j] += col_u[j]*x[i];
      }
    }
  }
  
  /* compute state.QP = 0.5*x'*(f+d);
             state.QD = 0.5*x'*(f-d); */
  for( i=0, state.QP = 0, state.QD=0; i < n; i++) 
  {
    state.QP += x[i]*(f[i]+d[i]);
    state.QD += x[i]*(f[i]-d[i]);
  }
  state.QP = 0.5*state.QP;
  state.QD = 0.5*state.QD;
  
  for( i=0; i < m; i++ ) 
  {
    for( j=0, tmp = LIBQP_PLUS_INF; j < nk[i]; j++ ) 
      tmp = LIBQP_MIN(tmp, d[inx[LIBQP_INDEX(j,i,n)]]);

    if(S[i] == 0) 
      state.QD += b[i]*tmp;
    else
      state.QD += b[i]*LIBQP_MIN(tmp,0);
  }
  
  /* print initial state */
  if( print_state != NULL) 
    print_state( state );

  /* ------------------------------------------------------------ 
    Main optimization loop 
  ------------------------------------------------------------ */
  while( state.exitflag == 100 ) 
  {
    state.nIter ++;

    /* go over blocks of variables coupled by lin. constraint */
    for( k=0; k < m; k++ ) 
    {       
        
      /* compute u = argmin_{i in I_k} d[i] 
             delta =  sum_{i in I_k} x[i]*d[i] - b*min_{i in I_k} */
      for( j=0, tmp = LIBQP_PLUS_INF, delta = 0; j < nk[k]; j++ ) 
      {
        i = inx[LIBQP_INDEX(j,k,n)];
        delta += x[i]*d[i];
        if( tmp > d[i] ) {
          tmp = d[i];
          u = i;
        }
      }

      if(S[k] != 0 && d[u] > 0) 
        u = -1;
      else
        delta -= b[k]*d[u];
            
      /* if satisfied then k-th block of variables needs update */
      if( delta > TolAbs/m && delta > TolRel*LIBQP_ABS(state.QP)/m) 
      {         
        /* for fixed u select v = argmax_{i in I_k} Improvement(i) */
        if( u != -1 ) 
        {
          col_u = (float64_t*)get_col(u);
          improv = -LIBQP_PLUS_INF;
          for( j=0; j < nk[k]; j++ ) 
          {
            i = inx[LIBQP_INDEX(j,k,n)];
           
            if(x[i] > 0 && i != uint32_t(u)) 
            {
              tmp_num = x[i]*(d[i] - d[u]); 
              tmp_den = x[i]*x[i]*(diag_H[u] - 2*col_u[i] + diag_H[i]);
              if( tmp_den > 0 ) 
              {
                if( tmp_num < tmp_den ) 
                  tmp = tmp_num*tmp_num / tmp_den;
                else 
                  tmp = tmp_num - 0.5 * tmp_den;
                 
                if( tmp > improv ) 
                { 
                  improv = tmp;
                  tau = LIBQP_MIN(1,tmp_num/tmp_den);
                  v = i;
                } 
              }
            }
          }

          /* check if virtual variable can be for updated */
          if(x_neq[k] > 0 && S[k] != 0) 
          {
            tmp_num = -x_neq[k]*d[u]; 
            tmp_den = x_neq[k]*x_neq[k]*diag_H[u];
            if( tmp_den > 0 ) 
            {
              if( tmp_num < tmp_den ) 
                tmp = tmp_num*tmp_num / tmp_den;
              else 
                tmp = tmp_num - 0.5 * tmp_den;
                 
              if( tmp > improv ) 
              { 
                improv = tmp;
                tau = LIBQP_MIN(1,tmp_num/tmp_den);
                v = -1;
              } 
            }
          }

          /* minimize objective w.r.t variable u and v */
          if(v != -1)
          {
            tmp = x[v]*tau;
            x[u] += tmp;
            x[v] -= tmp;

            /* update d = H*x + f */
            col_v = (float64_t*)get_col(v);
            for(i = 0; i < n; i++ )              
              d[i] += tmp*(col_u[i]-col_v[i]);
          }
          else
          {
            tmp = x_neq[k]*tau;
            x[u] += tmp;
            x_neq[k] -= tmp;

            /* update d = H*x + f */
            for(i = 0; i < n; i++ )              
              d[i] += tmp*col_u[i];
          }
        }
        else
        {
          improv = -LIBQP_PLUS_INF;
          for( j=0; j < nk[k]; j++ ) 
          {
            i = inx[LIBQP_INDEX(j,k,n)];
           
            if(x[i] > 0) 
            {
              tmp_num = x[i]*d[i]; 
              tmp_den = x[i]*x[i]*diag_H[i];
              if( tmp_den > 0 ) 
              {
                if( tmp_num < tmp_den ) 
                  tmp = tmp_num*tmp_num / tmp_den;
                else 
                  tmp = tmp_num - 0.5 * tmp_den;
                 
                if( tmp > improv ) 
                { 
                  improv = tmp;
                  tau = LIBQP_MIN(1,tmp_num/tmp_den);
                  v = i;
                } 
              }
            }
          }

          tmp = x[v]*tau;
          x_neq[k] += tmp;
          x[v] -= tmp;

          /* update d = H*x + f */
          col_v = (float64_t*)get_col(v);
          for(i = 0; i < n; i++ )              
            d[i] -= tmp*col_v[i];
        }

        /* update objective value */
        state.QP = state.QP - improv;
      }
    }
    
    /* Compute primal and dual objectives */
    for( i=0, state.QP = 0, state.QD=0; i < n; i++) 
    {
       state.QP += x[i]*(f[i]+d[i]);
       state.QD += x[i]*(f[i]-d[i]);
    }
    state.QP = 0.5*state.QP;
    state.QD = 0.5*state.QD;

    for( k=0; k < m; k++ ) 
    { 
      for( j=0,tmp = LIBQP_PLUS_INF; j < nk[k]; j++ ) {
        i = inx[LIBQP_INDEX(j,k,n)];
        tmp = LIBQP_MIN(tmp, d[i]);
      }
      
      if(S[k] == 0) 
        state.QD += b[k]*tmp;
      else
        state.QD += b[k]*LIBQP_MIN(tmp,0);
    }

    /* print state */
    if( print_state != NULL) 
      print_state( state );

    /* check stopping conditions */
    if(state.QP-state.QD <= LIBQP_ABS(state.QP)*TolRel ) state.exitflag = 1;
    else if( state.QP-state.QD <= TolAbs ) state.exitflag = 2;
    else if( state.QP <= QP_TH ) state.exitflag = 3;
    else if( state.nIter >= MaxIter) state.exitflag = 0;
  }

  /*----------------------------------------------------------
    Clean up
  ---------------------------------------------------------- */
cleanup:
  LIBQP_FREE( d );
  LIBQP_FREE( inx );
  LIBQP_FREE( nk );
  LIBQP_FREE( x_neq );
  
  return( state ); 
}
}

