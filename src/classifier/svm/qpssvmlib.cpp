/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Library of solvers for QP task required in StructSVM learning.
 *
 * Written (W) 1999-2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2008 Center for Machine Perception, CTU FEL Prague
--------------------------------------------------------------------
Synopsis:

  exitflag = qpssvm_solver( &get_col, diag_H, f, b, I, x, n, tmax,
             tolabs, tolrel, &t, &History, verb );

  exitflag = qpssvm_solver( &get_col, diag_H, f, b, I, x, n, tmax,
             tolabs, tolrel, &QP, &QD, verb );
Description:

 It solves the following QP task:

   min 0.5*x'*H*x + f'*x
    x

 subject to 

   sum(x(find(I==k))) <= b   for all k=1:max(I)
   x >= 0

 where I is a set of positive indices from (1 to max(I)).

 A precision of the found solution is given by the parameters tmax,
 tolabs and tolrel which define the stopping conditions:

 UB-LB <= tolabs      ->  exitflag = 1   Abs. tolerance.
 UB-LB <= UB*tolrel   ->  exitflag = 2   Relative tolerance.
 t >= tmax            ->  exitflag = 0   Number of iterations.

 UB ... Upper bound on the optimal solution, i.e., Q_P.
 LB ... Lower bound on the optimal solution, i.e., Q_D.
 t  ... Number of iterations.


Inputs/Outputs:

 const void* (*get_col)(uint32_t) retunr pointer to i-th column of H
 diag_H [double n x n] diagonal of H.
 f [double n x 1] is an arbitrary vector.
 b [double 1 x 1] scalar
 I [uint16_T n x 1] Indices (1..max(I)); max(I) <= n
 x [double n x 1] solution vector (inital solution).
 n [uint32_t 1 x 1] dimension of H.
 tmax [uint32_t 1 x 1] Max number of steps.
 tolrel [double 1 x 1] Relative tolerance.
 tolabs [double 1 x 1] Absolute tolerance.
 t [uint32_t 1 x 1] Number of iterations.
 History [double 2 x t] Value of LB and UB wrt. number of iterations.
 verb [int 1 x 1] if > 0 then prints info every verb-th iteation.

 For more info refer to TBA

 Modifications:
 01-Oct-2007, VF
 20-Feb-2006, VF
 18-feb-2006, VF

-------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#include "classifier/svm/libocas_common.h"
#include "classifier/svm/qpssvmlib.h"

/* --------------------------------------------------------------
 QPSSVM solver 

 Usage: exitflag = qpssvm_solver( &get_col, diag_H, f, b, I, x, n, tmax, 
         tolabs, tolrel, &QP, &QD, verb );   
-------------------------------------------------------------- */
int qpssvm_solver(const void* (*get_col)(uint32_t),
                  double *diag_H,
                  double *f,
                  double b,
                  uint16_t *I,
                  double *x,
                  uint32_t n,
                  uint32_t tmax,
                  double tolabs,
                  double tolrel,
                  double *QP,
                  double *QD,
				  int (*qpssvmlib_print)(const char *format, ...),
                  uint32_t verb)
{
  double *x_nequ;
  double *d;
  double *col_u, *col_v;
  double LB;
  double UB;
  double tmp;
  double improv;
  double tmp_num;
  double tmp_den=0;
  double tau=0;
  double delta;
  double yu;
  uint32_t *inx;
  uint32_t *nk;
  uint32_t m;
  uint32_t t;
  uint32_t u=0;
  uint32_t v=0;
  uint32_t k;
  uint32_t i, j;
  int exitflag;

  
  /* ------------------------------------------------------------ 
    Initialization                                               
  ------------------------------------------------------------ */

  x_nequ=NULL;
  inx=NULL;
  nk=NULL;
  d=NULL;

  /* count cumber of constraints */
  for( i=0, m=0; i < n; i++ ) m = MAX(m,I[i]);

  /* alloc and initialize x_nequ */
  x_nequ = (double*) OCAS_CALLOC(m, sizeof(double));
  if( x_nequ == NULL )
  {
	  exitflag=-2;
	  goto cleanup;
  }

  /* alloc Inx */
  inx = (uint32_t*) OCAS_CALLOC(m*n, sizeof(uint32_t));
  if( inx == NULL )
  {
	  exitflag=-2;
	  goto cleanup;
  }

  nk = (uint32_t*) OCAS_CALLOC(m, sizeof(uint32_t));
  if( nk == NULL )
  {
	  exitflag=-2;
	  goto cleanup;
  }

  for( i=0; i < m; i++ ) x_nequ[i] = b;
  for( i=0; i < n; i++ ) {
     k = I[i]-1;
     x_nequ[k] -= x[i];
     inx[INDEX2(nk[k],k,n)] = i;
     nk[k]++;
  }
    
  /* alloc d [n x 1] */
  d = (double*) OCAS_CALLOC(n, sizeof(double));
  if( d == NULL )
  {
	  exitflag=-2;
	  goto cleanup;
  }
 
  /* d = H*x + f; */
  for( i=0; i < n; i++ ) {
    if( x[i] > 0 ) {
      col_u = (double*)get_col(i);
      for( j=0; j < n; j++ ) {
          d[j] += col_u[j]*x[i];
      }
    }
  }
  for( i=0; i < n; i++ ) d[i] += f[i];
  
  /* UB = 0.5*x'*(f+d); */
  /* LB = 0.5*x'*(f-d); */
  for( i=0, UB = 0, LB=0; i < n; i++) {
    UB += x[i]*(f[i]+d[i]);
    LB += x[i]*(f[i]-d[i]);
  }
  UB = 0.5*UB;
  LB = 0.5*LB;

  /*
  for k=1:m,
    tmp = min(d(find(I==k)));
    if tmp < 0, LB = LB + b*tmp; end
  end
  */
  
  for( i=0; i < m; i++ ) {
    for( j=0, tmp = OCAS_PLUS_INF; j < nk[i]; j++ ) {
      tmp = MIN(tmp, d[inx[INDEX2(j,i,n)]]);
    }
    if( tmp < 0) LB += b*tmp;
  }
  
  exitflag = 0;
  t = 0;

  /* -- Main loop ---------------------------------------- */
  while( (exitflag == 0) && (t < tmax)) 
  {
    t++;

    exitflag = 1;
    for( k=0; k < m; k++ ) 
    {       
      /*
      inx = find(I==k);
      [tmp,u] = min(d(inx)); u = inx(u);
      */
        
     for( j=0, tmp = OCAS_PLUS_INF, delta = 0; j < nk[k]; j++ ) {
        i = inx[INDEX2(j,k,n)];
        delta += x[i]*d[i];
        if( tmp > d[i] ) {
          tmp = d[i];
          u = i;
        }
      }

      /* if d(u) < 0, yu = b; else yu = 0; end  */
      if( d[u] < 0) yu = b; else yu = 0;
     
      /* delta = x(inx)'*d(inx) - yu*d(u); */
      delta -= yu*d[u];
            
      if( delta > tolabs/m && delta > tolrel*ABS(UB)/m) 
      {
         exitflag = 0;
         
         if( yu > 0 ) 
         {
           col_u = (double*)get_col(u);      

           improv = -OCAS_PLUS_INF;
           for( j=0; j < nk[k]; j++ ) {
             i = inx[INDEX2(j,k,n)];
           
/*           for(i = 0; i < n; i++ ) {
             if( (I[i]-1 == k) && (i != u) && (x[i] > 0)) {              */
             if(x[i] > 0) {             
               
               tmp_num = x[i]*(d[i] - d[u]); 
               tmp_den = x[i]*x[i]*(diag_H[u] - 2*col_u[i] + diag_H[i]);
               if( tmp_den > 0 ) {
                 if( tmp_num < tmp_den ) {
                    tmp = tmp_num*tmp_num / tmp_den;
                 } else {
                    tmp = tmp_num - 0.5 * tmp_den;
                 }
               }
               if( tmp > improv ) {
                 improv = tmp;
                 tau = MIN(1,tmp_num/tmp_den);
                 v = i;
               }
             }
           }

           tmp_num = -x_nequ[k]*d[u];
           if( tmp_num > 0 ) {
             tmp_den = x_nequ[k]*x_nequ[k]*diag_H[u];
             if( tmp_den > 0 ) {
               if( tmp_num < tmp_den ) {
                 tmp = tmp_num*tmp_num / tmp_den;
               } else {
                   tmp = tmp_num - 0.5 * tmp_den;
               }
             }
           } else {
             tmp = -OCAS_PLUS_INF; 
           }
           
           if( tmp > improv ) {
              tau = MIN(1,tmp_num/tmp_den);
              for( i = 0; i < n; i++ ) {             
                d[i] += x_nequ[k]*tau*col_u[i];
              }
             x[u] += tau*x_nequ[k];
             x_nequ[k] -= tau*x_nequ[k];
               
           } else {
            
             /* updating with the best line segment */
             col_v = (double*)get_col(v);
             for( i = 0; i < n; i++ ) {             
               d[i] += x[v]*tau*(col_u[i]-col_v[i]);
             }

             x[u] += tau*x[v];
             x[v] -= tau*x[v];
           }
         }
         else
         {
           improv = -OCAS_PLUS_INF;
           for( j=0; j < nk[k]; j++ ) {
             i = inx[INDEX2(j,k,n)];
           
/*           for(i = 0; i < n; i++ ) {
             if( (I[i]-1 == k) && (x[i] > 0)) {*/
             if( x[i] > 0 && d[i] > 0) {
                
               tmp_num = x[i]*d[i]; 
               tmp_den = x[i]*x[i]*diag_H[i];
               if( tmp_den > 0 ) {
                 if( tmp_num < tmp_den ) {
                    tmp = tmp_num*tmp_num / tmp_den;
                 } else {
                    tmp = tmp_num - 0.5 * tmp_den;
                 }
               }
               if( tmp > improv ) {
                 improv = tmp;
                 tau = MIN(1,tmp_num/tmp_den);
                 v = i;
               }
             }    
           }

           /* updating with the best line segment */
           col_v = (double*)get_col(v);
           for( i = 0; i < n; i++ ) {             
             d[i] -= x[v]*tau*col_v[i];
           }

           x_nequ[k] += tau*x[v];
           x[v] -= tau*x[v];         
         }

         UB = UB - improv;
      }
                   
    }

    /* -- Computing LB --------------------------------------*/

    /*
    LB = 0.5*x'*(f-d);   
    for k=1:n,
      LB = LB + b*min(d(find(I==k)));
    end */
    
    for( i=0, UB = 0, LB=0; i < n; i++) {
       UB += x[i]*(f[i]+d[i]);
       LB += x[i]*(f[i]-d[i]);
    }
    UB = 0.5*UB;
    LB = 0.5*LB;

    for( k=0; k < m; k++ ) { 
      for( j=0,tmp = OCAS_PLUS_INF; j < nk[k]; j++ ) {
        i = inx[INDEX2(j,k,n)];

        tmp = MIN(tmp, d[i]);
      }
      if( tmp < 0) LB += b*tmp;
    }

    if( verb > 0 && (exitflag > 0 || (t % verb)==0 )) {
       qpssvmlib_print("%d: UB=%.10f, LB=%.10f, UB-LB=%.10f, (UB-LB)/|UB|=%.10f \n",
        t, UB, LB, UB-LB, (UB!=0) ? (UB-LB)/ABS(UB) : 0);      
    }    

  }

  /* -- Find which stopping consition has been used -------- */
  if( UB-LB < tolabs ) exitflag = 1;
  else if(UB-LB < ABS(UB)*tolrel ) exitflag = 2;
  else exitflag = 0;

  /*----------------------------------------------------------   
    Set up outputs                                          
  ---------------------------------------------------------- */
  *QP = UB;
  *QD = LB;

  /*----------------------------------------------------------
    Clean up
  ---------------------------------------------------------- */
cleanup:
  OCAS_FREE( d );
  OCAS_FREE( inx );
  OCAS_FREE( nk );
  OCAS_FREE( x_nequ );  
  
  return( exitflag ); 

}

