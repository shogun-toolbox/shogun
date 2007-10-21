/*-----------------------------------------------------------------------
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Library of solvers for QP task required in StructSVM learning.
 * 
 * Description:
 *  
 *  It solves the following QP task:
 *   
 *    min 0.5*x'*H*x + f'*x
 *     x
 * 
 *  subject to 
 *  
 *    sum(x(find(I==k))) <= b   for all k=1:max(I)
 *    x >= 0
 * 
 *  where I is a set of positive indices from (1 to max(I)).
 * 
 *  A precision of the found solution is given by the parameters tmax, 
 *  tolabs and tolrel which define the stopping conditions:
 *  
 *  UB-LB <= tolabs      ->  exitflag = 1   Abs. tolerance.
 *  UB-LB <= UB*tolrel   ->  exitflag = 2   Relative tolerance.
 *  t >= tmax            ->  exitflag = 0   Number of iterations.
 * 
 *  UB ... Upper bound on the optimal solution, i.e., Q_P.
 *  LB ... Lower bound on the optimal solution, i.e., Q_D.
 *  t  ... Number of iterations.
 * 
 * 
 * Inputs/Outputs:
 * 
 *  const void* (*get_col)(INT) retunr poINTer to i-th column of H
 *  diag_H [DREAL n x n] diagonal of H.
 *  f [DREAL n x 1] is an arbitrary vector.
 *  b [DREAL 1 x 1] scalar
 *  I [uINT16_T n x 1] Indices (1..max(I)); max(I) <= n
 *  x [DREAL n x 1] solution vector (inital solution).
 *  n [INT 1 x 1] dimension of H.
 *  tmax [INT 1 x 1] Max number of steps.
 *  tolrel [DREAL 1 x 1] Relative tolerance.
 *  tolabs [DREAL 1 x 1] Absolute tolerance.
 *  t [INT 1 x 1] Number of iterations.
 *  History [DREAL 2 x t] Value of LB and UB wrt. number of iterations.
 *  verb [INT 1 x 1] if > 0 then prints info every verb-th iteation.
 * 
 *  For more info refer to TBA
 * 
 *  Modifications:
 *  01-Oct-2007, VF
 *  20-Feb-2006, VF
 *  18-feb-2006, VF
-------------------------------------------------------------------- */
#include "classifier/svm/qpssvmlib.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define HISTORY_BUF 1000000

#define MINUS_INF INT_MIN
#define PLUS_INF  INT_MAX

#define ABS(A) (((A) >= 0) ? (A) : (-A))
#define MIN(A,B) (((A) < (B)) ? (A) : (B))
#define MAX(A,B) (((A) > (B)) ? (A) : (B))
#define INDEX(ROW,COL,DIM) ((COL*DIM)+ROW)


CQPSSVMLib::CQPSSVMLib(DREAL* H, WORD* I, DREAL* f, INT n, INT m)
{
	ASSERT(H && n>0);
	m_H=H;
	m_dim = n;
	m_diag_H=NULL; 
	m_f=f;
	m_I=I;
	//m_UB=UB;
	//m_tmax = INT_MAX;
	//m_tolabs = 0;
	//m_tolrel = 1e-6;
	//m_tolKKT = 0;
	//m_solver = QPB_SOLVER_SCA;
}

CQPSSVMLib::~CQPSSVMLib()
{
	delete[] m_diag_H;
}


/* --------------------------------------------------------------
 QPSSVM solver 

 Usage: exitflag = qpssvm_imdm( &get_col, diag_H, f, b, I, x, n, tmax, 
         tolabs, tolrel, &t, &History, verb );   
-------------------------------------------------------------- */
INT CQPSSVMLib::qpssvm_imdm(
                  DREAL *x,
                  INT *ptr_t,
                  DREAL **ptr_History)
{
  DREAL *x_nequ;
  DREAL *d;
  DREAL *History;
  DREAL *col_u, *col_v;
  DREAL *tmp_ptr;
  DREAL LB;
  DREAL UB;
  DREAL tmp;
  DREAL improv;
  DREAL tmp_num;
  DREAL tmp_den;
  DREAL tau;
  DREAL delta;
  DREAL yu;
  INT *inx;
  INT *nk;
  INT m;
  INT t;
  INT u;
  INT v;
  INT k;
  INT i, j;
  INT History_size;
  INT exitflag;

  
  /* ------------------------------------------------------------ 
    Initialization                                               
  ------------------------------------------------------------ */

  /* count cumber of constraints */
  for( i=0, m=0; i < m_var; i++ ) m = MAX(m,m_I[i]);

  /* alloc and initialize x_nequ */
  x_nequ = (DREAL*) calloc(m, sizeof(DREAL));
  if( x_nequ == NULL ) SG_ERROR("Not enough memory.");

  /* alloc Inx */
  inx = (INT*) calloc(m*m_var, sizeof(INT));
  if( inx == NULL ) SG_ERROR("Not enough memory.");

  nk = (INT*) calloc(m, sizeof(INT));
  if( nk == NULL ) SG_ERROR("Not enough memory.");

  for( i=0; i < m; i++ ) x_nequ[i] = m_b;
  for( i=0; i < m_var; i++ ) {
     k = m_I[i]-1;
     x_nequ[k] -= x[i];
     inx[INDEX(nk[k],k,m_var)] = i;
     nk[k]++;
  }
    
  /* alloc History [2 x HISTORY_BUF] */
  History_size = (m_tmax < HISTORY_BUF ) ? m_tmax+1 : HISTORY_BUF;
  History = (DREAL*) calloc(History_size*2,sizeof(DREAL));
  if( History == NULL ) SG_ERROR("Not enough memory.");

  /* alloc d [n x 1] */
  d = (DREAL*) calloc(m_var, sizeof(DREAL));
  if( d == NULL ) SG_ERROR("Not enough memory.");
 
  /* d = H*x + f; */
  for( i=0; i < m_var; i++ ) {
    if( x[i] > 0 ) {
      col_u = (DREAL*)get_col(i);
      for( j=0; j < m_var; j++ ) {
          d[j] += col_u[j]*x[i];
      }
    }
  }
  for( i=0; i < m_var; i++ ) d[i] += m_f[i];
  
  /* UB = 0.5*x'*(f+d); */
  /* LB = 0.5*x'*(f-d); */
  for( i=0, UB = 0, LB=0; i < m_var; i++) {
    UB += x[i]*(m_f[i]+d[i]);
    LB += x[i]*(m_f[i]-d[i]);
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
    for( j=0, tmp = PLUS_INF; j < nk[i]; j++ ) {
      tmp = MIN(tmp, d[inx[INDEX(j,i,m_var)]]);
    }
    if( tmp < 0) LB += m_b*tmp;
  }
  
  /*
  for( i=0; i < m; i++ ) {
    for( j=0, tmp = PLUS_INF; j < m_var; j++ ) {
      if( I[j]-1 == i ) tmp = MIN(tmp, d[j]);
    }
    if( tmp < 0) LB += b*tmp;
  }*/

  exitflag = 0;
  t = 0;
  History[INDEX(0,0,2)] = LB;
  History[INDEX(1,0,2)] = UB;

  /* -- Main loop ---------------------------------------- */
  while( (exitflag == 0) && (t < m_tmax)) 
  {
    t++;

    exitflag = 1;
    for( k=0; k < m; k++ ) 
    {       
      /*
      inx = find(I==k);
      [tmp,u] = min(d(inx)); u = inx(u);
      */
        
     for( j=0,tmp = PLUS_INF, delta = 0; j < nk[k]; j++ ) {
        i = inx[INDEX(j,k,m_var)];
/*      for( i=0, tmp = PLUS_INF, delta = 0; i < m_var; i++ ) {
        if( I[i]-1 == k) {*/
        delta += x[i]*d[i];
        if( tmp > d[i] ) {
          tmp = d[i];
          u = i;
        }
      }

      /* if d(u) < 0, yu = b; else yu = 0; end  */
      if( d[u] < 0) yu = m_b; else yu = 0;
     
      /* delta = x(inx)'*d(inx) - yu*d(u); */
      delta -= yu*d[u];
            
      if( delta > m_tolabs/m && delta > m_tolrel*ABS(UB)/m) 
      {
         exitflag = 0;
         
         if( yu > 0 ) 
         {
           col_u = (DREAL*)get_col(u);      

           improv = MINUS_INF;
           for( j=0; j < nk[k]; j++ ) {
             i = inx[INDEX(j,k,m_var)];
           
/*           for(i = 0; i < m_var; i++ ) {
             if( (I[i]-1 == k) && (i != u) && (x[i] > 0)) {              */
             if(x[i] > 0) {             
               
               tmp_num = x[i]*(d[i] - d[u]); 
               tmp_den = x[i]*x[i]*(m_diag_H[u] - 2*col_u[i] + m_diag_H[i]);
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
             tmp_den = x_nequ[k]*x_nequ[k]*m_diag_H[u];
             if( tmp_den > 0 ) {
               if( tmp_num < tmp_den ) {
                 tmp = tmp_num*tmp_num / tmp_den;
               } else {
                   tmp = tmp_num - 0.5 * tmp_den;
               }
             }
           } else {
             tmp = MINUS_INF; 
           }
           
           if( tmp > improv ) {
              tau = MIN(1,tmp_num/tmp_den);
              for( i = 0; i < m_var; i++ ) {             
                d[i] += x_nequ[k]*tau*col_u[i];
              }
             x[u] += tau*x_nequ[k];
             x_nequ[k] -= tau*x_nequ[k];
               
           } else {
            
             /* updating with the best line segment */
             col_v = (DREAL*)get_col(v);
             for( i = 0; i < m_var; i++ ) {             
               d[i] += x[v]*tau*(col_u[i]-col_v[i]);
             }

             x[u] += tau*x[v];
             x[v] -= tau*x[v];
           }
         }
         else
         {
           improv = MINUS_INF;
           for( j=0; j < nk[k]; j++ ) {
             i = inx[INDEX(j,k,m_var)];
           
/*           for(i = 0; i < m_var; i++ ) {
             if( (I[i]-1 == k) && (x[i] > 0)) {*/
             if( x[i] > 0 && d[i] > 0) {
                
               tmp_num = x[i]*d[i]; 
               tmp_den = x[i]*x[i]*m_diag_H[i];
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
           col_v = (DREAL*)get_col(v);
           for( i = 0; i < m_var; i++ ) {             
             d[i] -= x[v]*tau*col_v[i];
           }

           x_nequ[k] += tau*x[v];
           x[v] -= tau*x[v];         
         }
                    
/*         for( i=0, UB = 0; i < m_var; i++) {
            UB += x[i]*(f[i]+d[i]);
         }
         UB = 0.5*UB;
 */
         UB = UB - improv;
      }
                   
      /* SG_PRINT("t=%d,k=%d, u=%d, tau1=%f, den1=%f, num1=%f, delta=%f\n", 
             t,k,u,tau1,den1,num1,delta);*/

    }

    /* -- Computing LB --------------------------------------*/

    /*
    LB = 0.5*x'*(f-d);   
    for k=1:n,
      LB = LB + b*min(d(find(I==k)));
    end */
    
    for( i=0, UB = 0, LB=0; i < m_var; i++) {
       UB += x[i]*(m_f[i]+d[i]);
       LB += x[i]*(m_f[i]-d[i]);
    }
    UB = 0.5*UB;
    LB = 0.5*LB;

    for( k=0; k < m; k++ ) { 
      for( j=0,tmp = PLUS_INF; j < nk[k]; j++ ) {
        i = inx[INDEX(j,k,m_var)];

/*      for( j=0, tmp = PLUS_INF; j < m_var; j++ ) {
        if( I[j]-1 == i ) tmp = MIN(tmp, d[j]);*/
        tmp = MIN(tmp, d[i]);
      }
      if( tmp < 0) LB += m_b*tmp;
    }

    /* Store LB and UB */
    if( t < History_size ) {
      History[INDEX(0,t,2)] = LB;
      History[INDEX(1,t,2)] = UB;
    }
    else {
      tmp_ptr = (DREAL*) calloc((History_size+HISTORY_BUF)*2,sizeof(DREAL));
      if( tmp_ptr == NULL ) SG_ERROR("Not enough memory.");
      for( i = 0; i < History_size; i++ ) {
        tmp_ptr[INDEX(0,i,2)] = History[INDEX(0,i,2)];
        tmp_ptr[INDEX(1,i,2)] = History[INDEX(1,i,2)];
      }
      tmp_ptr[INDEX(0,t,2)] = LB;
      tmp_ptr[INDEX(1,t,2)] = UB;
  
      History_size += HISTORY_BUF;
      free( History );
      History = tmp_ptr;
    }

    if( m_verb > 0 && (exitflag > 0 || (t % m_verb)==0 )) {
       SG_PRINT("%d: UB=%.10f, LB=%.10f, UB-LB=%.10f, (UB-LB)/|UB|=%.10f \n",
        t, UB, LB, UB-LB, (UB!=0) ? (UB-LB)/ABS(UB) : 0);      
    }    

  }

  /* -- Find which stopping consition has been used -------- */
  if( UB-LB < m_tolabs ) exitflag = 1;
  else if(UB-LB < ABS(UB)*m_tolrel ) exitflag = 2;
  else exitflag = 0;

  /*----------------------------------------------------------   
    Set up outputs                                          
  ---------------------------------------------------------- */
  (*ptr_t) = t;
  (*ptr_History) = History;

  /*----------------------------------------------------------
    Clean up
  ---------------------------------------------------------- */
  free( d );
  free( inx );
  free( nk );
  
  return( exitflag ); 

}



/* --------------------------------------------------------------
 QPSSVM solver 

 Usage: exitflag = qpssvm_solver( &get_col, diag_H, f, b, I, x, n, tmax, 
         tolabs, tolrel, &QP, &QD, verb );   
-------------------------------------------------------------- */
INT CQPSSVMLib::qpssvm_solver(DREAL *x)
{
  DREAL *x_nequ;
  DREAL *d;
  DREAL *col_u, *col_v;
  DREAL LB;
  DREAL UB;
  DREAL tmp;
  DREAL improv;
  DREAL tmp_num;
  DREAL tmp_den=0;
  DREAL tau=0.0;
  DREAL delta;
  DREAL yu;
  INT *inx;
  INT *nk;
  INT m;
  INT t;
  INT u=0;
  INT v=0;
  INT k;
  INT i, j;
  INT exitflag;

  
  /* ------------------------------------------------------------ 
    Initialization                                               
  ------------------------------------------------------------ */

  /* count cumber of constraints */
  for( i=0, m=0; i < m_var; i++ ) m = MAX(m,m_I[i]);

  /* alloc and initialize x_nequ */
  x_nequ = (DREAL*) calloc(m, sizeof(DREAL));
  if( x_nequ == NULL ) SG_ERROR("Not enough memory.");

  /* alloc Inx */
  inx = (INT*) calloc(m*m_var, sizeof(INT));
  if( inx == NULL ) SG_ERROR("Not enough memory.");

  nk = (INT*) calloc(m, sizeof(INT));
  if( nk == NULL ) SG_ERROR("Not enough memory.");

  for( i=0; i < m; i++ ) x_nequ[i] = m_b;
  for( i=0; i < m_var; i++ ) {
     k = m_I[i]-1;
     x_nequ[k] -= x[i];
     inx[INDEX(nk[k],k,m_var)] = i;
     nk[k]++;
  }
    
  /* alloc d [n x 1] */
  d = (DREAL*) calloc(m_var, sizeof(DREAL));
  if( d == NULL ) SG_ERROR("Not enough memory.");
 
  /* d = H*x + f; */
  for( i=0; i < m_var; i++ ) {
    if( x[i] > 0 ) {
      col_u = (DREAL*)get_col(i);
      for( j=0; j < m_var; j++ ) {
          d[j] += col_u[j]*x[i];
      }
    }
  }
  for( i=0; i < m_var; i++ ) d[i] += m_f[i];
  
  /* UB = 0.5*x'*(f+d); */
  /* LB = 0.5*x'*(f-d); */
  for( i=0, UB = 0, LB=0; i < m_var; i++) {
    UB += x[i]*(m_f[i]+d[i]);
    LB += x[i]*(m_f[i]-d[i]);
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
    for( j=0, tmp = PLUS_INF; j < nk[i]; j++ ) {
      tmp = MIN(tmp, d[inx[INDEX(j,i,m_var)]]);
    }
    if( tmp < 0) LB += m_b*tmp;
  }
  
  /*
  for( i=0; i < m; i++ ) {
    for( j=0, tmp = PLUS_INF; j < m_var; j++ ) {
      if( I[j]-1 == i ) tmp = MIN(tmp, d[j]);
    }
    if( tmp < 0) LB += b*tmp;
  }*/

  exitflag = 0;
  t = 0;

  /* -- Main loop ---------------------------------------- */
  while( (exitflag == 0) && (t < m_tmax)) 
  {
    t++;

    exitflag = 1;
    for( k=0; k < m; k++ ) 
    {       
      /*
      inx = find(I==k);
      [tmp,u] = min(d(inx)); u = inx(u);
      */
        
     for( j=0,tmp = PLUS_INF, delta = 0; j < nk[k]; j++ ) {
        i = inx[INDEX(j,k,m_var)];
/*      for( i=0, tmp = PLUS_INF, delta = 0; i < m_var; i++ ) {
        if( I[i]-1 == k) {*/
        delta += x[i]*d[i];
        if( tmp > d[i] ) {
          tmp = d[i];
          u = i;
        }
      }

      /* if d(u) < 0, yu = b; else yu = 0; end  */
      if( d[u] < 0) yu = m_b; else yu = 0;
     
      /* delta = x(inx)'*d(inx) - yu*d(u); */
      delta -= yu*d[u];
            
      if( delta > m_tolabs/m && delta > m_tolrel*ABS(UB)/m) 
      {
         exitflag = 0;
         
         if( yu > 0 ) 
         {
           col_u = (DREAL*)get_col(u);      

           improv = MINUS_INF;
           for( j=0; j < nk[k]; j++ ) {
             i = inx[INDEX(j,k,m_var)];
           
/*           for(i = 0; i < m_var; i++ ) {
             if( (I[i]-1 == k) && (i != u) && (x[i] > 0)) {              */
             if(x[i] > 0) {             
               
               tmp_num = x[i]*(d[i] - d[u]); 
               tmp_den = x[i]*x[i]*(m_diag_H[u] - 2*col_u[i] + m_diag_H[i]);
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
             tmp_den = x_nequ[k]*x_nequ[k]*m_diag_H[u];
             if( tmp_den > 0 ) {
               if( tmp_num < tmp_den ) {
                 tmp = tmp_num*tmp_num / tmp_den;
               } else {
                   tmp = tmp_num - 0.5 * tmp_den;
               }
             }
           } else {
             tmp = MINUS_INF; 
           }
           
           if( tmp > improv ) {
              tau = MIN(1,tmp_num/tmp_den);
              for( i = 0; i < m_var; i++ ) {             
                d[i] += x_nequ[k]*tau*col_u[i];
              }
             x[u] += tau*x_nequ[k];
             x_nequ[k] -= tau*x_nequ[k];
               
           } else {
            
             /* updating with the best line segment */
             col_v = (DREAL*)get_col(v);
             for( i = 0; i < m_var; i++ ) {             
               d[i] += x[v]*tau*(col_u[i]-col_v[i]);
             }

             x[u] += tau*x[v];
             x[v] -= tau*x[v];
           }
         }
         else
         {
           improv = MINUS_INF;
           for( j=0; j < nk[k]; j++ ) {
             i = inx[INDEX(j,k,m_var)];
           
/*           for(i = 0; i < m_var; i++ ) {
             if( (I[i]-1 == k) && (x[i] > 0)) {*/
             if( x[i] > 0 && d[i] > 0) {
                
               tmp_num = x[i]*d[i]; 
               tmp_den = x[i]*x[i]*m_diag_H[i];
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
           col_v = (DREAL*)get_col(v);
           for( i = 0; i < m_var; i++ ) {             
             d[i] -= x[v]*tau*col_v[i];
           }

           x_nequ[k] += tau*x[v];
           x[v] -= tau*x[v];         
         }
                    
/*         for( i=0, UB = 0; i < m_var; i++) {
            UB += x[i]*(f[i]+d[i]);
         }
         UB = 0.5*UB;
 */
         UB = UB - improv;
      }
                   
      /* SG_PRINT("t=%d,k=%d, u=%d, tau1=%f, den1=%f, num1=%f, delta=%f\n", 
             t,k,u,tau1,den1,num1,delta);*/

    }

    /* -- Computing LB --------------------------------------*/

    /*
    LB = 0.5*x'*(f-d);   
    for k=1:n,
      LB = LB + b*min(d(find(I==k)));
    end */
    
    for( i=0, UB = 0, LB=0; i < m_var; i++) {
       UB += x[i]*(m_f[i]+d[i]);
       LB += x[i]*(m_f[i]-d[i]);
    }
    UB = 0.5*UB;
    LB = 0.5*LB;

    for( k=0; k < m; k++ ) { 
      for( j=0,tmp = PLUS_INF; j < nk[k]; j++ ) {
        i = inx[INDEX(j,k,m_var)];

/*      for( j=0, tmp = PLUS_INF; j < m_var; j++ ) {
        if( I[j]-1 == i ) tmp = MIN(tmp, d[j]);*/
        tmp = MIN(tmp, d[i]);
      }
      if( tmp < 0) LB += m_b*tmp;
    }

    if( m_verb > 0 && (exitflag > 0 || (t % m_verb)==0 )) {
       SG_PRINT("%d: UB=%.10f, LB=%.10f, UB-LB=%.10f, (UB-LB)/|UB|=%.10f \n",
        t, UB, LB, UB-LB, (UB!=0) ? (UB-LB)/ABS(UB) : 0);      
    }    

  }

  /* -- Find which stopping consition has been used -------- */
  if( UB-LB < m_tolabs ) exitflag = 1;
  else if(UB-LB < ABS(UB)*m_tolrel ) exitflag = 2;
  else exitflag = 0;

  /*----------------------------------------------------------   
    Set up outputs                                          
  ---------------------------------------------------------- */
  m_QP = UB;
  m_QD = LB;

  /*----------------------------------------------------------
    Clean up
  ---------------------------------------------------------- */
  free( d );
  free( inx );
  free( nk );
  
  return( exitflag ); 

}

