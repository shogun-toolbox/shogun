/*-----------------------------------------------------------------------
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Library of solvers for Generalized Nearest Point Problem (GNPP).
 *
 * Written (W) 1999-2007 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2007 Center for Machine Perception, CTU FEL Prague 
 *
-------------------------------------------------------------------- */

#include <math.h>
#include <limits.h>
#include "lib/common.h"
#include "lib/io.h"
#include "lib/Mathematics.h"

#include "classifier/svm/gnpplib.h"
#include "kernel/Kernel.h"

#define HISTORY_BUF 1000000

#define MINUS_INF INT_MIN
#define PLUS_INF  INT_MAX

#define INDEX(ROW,COL,DIM) ((COL*DIM)+ROW)


CGNPPLib::CGNPPLib(DREAL* vector_y, CKernel* kernel, INT num_data, DREAL reg_const) : CSGObject()
{
  m_reg_const = reg_const;
  m_num_data = num_data;
  m_vector_y = vector_y;
  m_kernel = kernel;

  Cache_Size = ((LONG) kernel->get_cache_size())*1024*1024/(sizeof(DREAL)*num_data);
  Cache_Size = CMath::min(Cache_Size, (LONG) num_data);

  SG_INFO("using %d kernel cache lines\n", Cache_Size);
  ASSERT(Cache_Size > 2);

  /* allocates memory for kernel cache */
  kernel_columns = new DREAL*[Cache_Size];
  ASSERT(kernel_columns);
  cache_index = new DREAL[Cache_Size];
  ASSERT(cache_index);

  for(INT i = 0; i < Cache_Size; i++ ) 
  {
    kernel_columns[i] = new DREAL[num_data];
    ASSERT(kernel_columns[i]);
    cache_index[i] = -2;
  }
  first_kernel_inx = 0;

}

CGNPPLib::~CGNPPLib()
{
  for(INT i = 0; i < Cache_Size; i++ ) 
      delete[] kernel_columns[i];

  delete[] cache_index;
  delete[] kernel_columns;
}

/* --------------------------------------------------------------
 QP solver based on Mitchell-Demyanov-Malozemov  algorithm.

 Usage: exitflag = gnpp_mdm( diag_H, vector_c, vector_y,
       dim, tmax, tolabs, tolrel, th, &alpha, &t, &aHa11, &aHa22, &History );
-------------------------------------------------------------- */
int CGNPPLib::gnpp_mdm(double *diag_H,
                       double *vector_c,
                       double *vector_y,
                       INT dim, 
                       INT tmax,
                       double tolabs,
                       double tolrel,
                       double th,
                       double *alpha,
                       INT  *ptr_t, 
                       double *ptr_aHa11,
                       double *ptr_aHa22,
                       double **ptr_History,
                       INT verb)
{
  double LB;
  double UB;
  double aHa11, aHa12, aHa22, ac1, ac2;
  double tmp;
  double Huu, Huv, Hvv;
  double min_beta1, max_beta1, min_beta2, max_beta2, beta;
  double lambda;
  double delta1, delta2;
  double *History;
  double *Ha1;
  double *Ha2;
  double *tmp_ptr;
  double *col_u, *col_v;
  double *col_v1, *col_v2;
  long u1=0, u2=0;
  long v1, v2;
  long i;
  long t;
  long History_size;
  int exitflag;

  /* ------------------------------------------------------------ */
  /* Initialization                                               */
  /* ------------------------------------------------------------ */

  Ha1 = new DREAL[dim];
  if( Ha1 == NULL ) SG_ERROR("Not enough memory.\n");
  Ha2 = new DREAL[dim];
  if( Ha2 == NULL ) SG_ERROR("Not enough memory.\n");

  History_size = (tmax < HISTORY_BUF ) ? tmax+1 : HISTORY_BUF;
  History = new DREAL[History_size*2];
  if( History == NULL ) SG_ERROR("Not enough memory.\n");

  /* inx1 = firts of find( y ==1 ), inx2 = firts of find( y ==2 ) */
  v1 = -1; v2 = -1; i = 0;
  while( (v1 == -1 || v2 == -1) && i < dim ) {
    if( v1 == -1 && vector_y[i] == 1 ) { v1 = i; }
    if( v2 == -1 && vector_y[i] == 2 ) { v2 = i; } 
    i++;
  }

  col_v1 = (double*)get_col(v1,-1);
  col_v2 = (double*)get_col(v2,v1);
  
  aHa12 = col_v1[v2];
  aHa11 = diag_H[v1];
  aHa22 = diag_H[v2];
  ac1 = vector_c[v1];
  ac2 = vector_c[v2];

  min_beta1 = PLUS_INF; min_beta2 = PLUS_INF;
  for( i = 0; i < dim; i++ ) 
  {
    alpha[i] = 0;
    Ha1[i] = col_v1[i];
    Ha2[i] = col_v2[i];

    beta = Ha1[i] + Ha2[i] + vector_c[i];

    if( vector_y[i] == 1 && min_beta1 > beta ) {
      u1 = i;
      min_beta1 = beta;
    }

    if( vector_y[i] == 2 && min_beta2 > beta ) {
      u2 = i;
      min_beta2 = beta;
    }
  }

  alpha[v1] = 1;
  alpha[v2] = 1;

  UB = 0.5*(aHa11 + 2*aHa12 + aHa22) + ac1 + ac2;
  LB = min_beta1 + min_beta2 - 0.5*(aHa11 + 2*aHa12 + aHa22);

  delta1 = Ha1[v1] + Ha2[v1] + vector_c[v1] - min_beta1;
  delta2 = Ha1[v2] + Ha2[v2] + vector_c[v2] - min_beta2;

  t = 0;
  History[INDEX(0,0,2)] = LB;
  History[INDEX(1,0,2)] = UB;

  if( verb ) {
    SG_PRINT("Init: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f \n",
      UB, LB, UB-LB,(UB-LB)/UB);
  }  

  /* Stopping conditions */
  if( UB-LB <= tolabs ) exitflag = 1;
  else if(UB-LB <= CMath::abs(UB)*tolrel ) exitflag = 2;
  else if(LB > th) exitflag = 3;
  else exitflag = -1;

  /* ------------------------------------------------------------ */
  /* Main optimization loop                                       */
  /* ------------------------------------------------------------ */

  while( exitflag == -1 ) 
  {
    t++;     

    if( delta1 > delta2 ) 
    {
      col_u = (double*)get_col(u1,-1);
      col_v = (double*)get_col(v1,u1);

      Huu = diag_H[u1];
      Hvv = diag_H[v1];
      Huv = col_u[v1];

      lambda = delta1/(alpha[v1]*(Huu - 2*Huv + Hvv ));
      lambda = CMath::min(1.0,lambda);

      tmp = lambda*alpha[v1];

      aHa11 = aHa11 + 2*tmp*(Ha1[u1]-Ha1[v1])+tmp*tmp*( Huu - 2*Huv + Hvv );
      aHa12 = aHa12 + tmp*(Ha2[u1]-Ha2[v1]);
      ac1 = ac1 + tmp*(vector_c[u1]-vector_c[v1]);

      alpha[u1] = alpha[u1] + tmp;
      alpha[v1] = alpha[v1] - tmp;

      min_beta1 = PLUS_INF; min_beta2 = PLUS_INF;
      max_beta1 = MINUS_INF; max_beta2 = MINUS_INF; 
      for( i = 0; i < dim; i ++ )
      {
         Ha1[i] = Ha1[i] + tmp*(col_u[i] - col_v[i]);

         beta = Ha1[i] + Ha2[i] + vector_c[i];
         if( vector_y[i] == 1 ) 
           {
             if( min_beta1 > beta ) { u1 = i; min_beta1 = beta; }
             if( max_beta1 < beta && alpha[i] > 0 ) { v1 = i; max_beta1 = beta; }
           }
         else
           {
             if( min_beta2 > beta ) { u2 = i; min_beta2 = beta; }
             if( max_beta2 < beta && alpha[i] > 0) { v2 = i; max_beta2 = beta; }
           }
      }
    }
    else
    {
      col_u = (double*)get_col(u2,-1);
      col_v = (double*)get_col(v2,u2);

      Huu = diag_H[u2];
      Hvv = diag_H[v2];
      Huv = col_u[v2];
  
      lambda = delta2/(alpha[v2]*( Huu - 2*Huv + Hvv ));
      lambda = CMath::min(1.0,lambda);

      tmp = lambda*alpha[v2];
      aHa22 = aHa22 + 2*tmp*( Ha2[u2]-Ha2[v2]) + tmp*tmp*( Huu - 2*Huv + Hvv);
      aHa12 = aHa12 + tmp*(Ha1[u2]-Ha1[v2]);
      ac2 = ac2 + tmp*( vector_c[u2]-vector_c[v2] );

      alpha[u2] = alpha[u2] + tmp;
      alpha[v2] = alpha[v2] - tmp;

      min_beta1 = PLUS_INF; min_beta2 = PLUS_INF;
      max_beta1 = MINUS_INF; max_beta2 = MINUS_INF; 
      for(i = 0; i < dim; i++ ) 
      {  
         Ha2[i] = Ha2[i] + tmp*( col_u[i] - col_v[i] );

         beta = Ha1[i] + Ha2[i] + vector_c[i];

         if( vector_y[i] == 1 ) 
         {
           if( min_beta1 > beta ) { u1 = i; min_beta1 = beta; }
           if( max_beta1 < beta && alpha[i] > 0 ) { v1 = i; max_beta1 = beta; }
         }
         else
         {
           if( min_beta2 > beta ) { u2 = i; min_beta2 = beta; }
           if( max_beta2 < beta && alpha[i] > 0) { v2 = i; max_beta2 = beta; }
         }
      }
    }

    UB = 0.5*(aHa11 + 2*aHa12 + aHa22) + ac1 + ac2;
    LB = min_beta1 + min_beta2 - 0.5*(aHa11 + 2*aHa12 + aHa22);
  
    delta1 = Ha1[v1] + Ha2[v1] + vector_c[v1] - min_beta1;
    delta2 = Ha1[v2] + Ha2[v2] + vector_c[v2] - min_beta2;

    /* Stopping conditions */
    if( UB-LB <= tolabs ) exitflag = 1; 
    else if( UB-LB <= CMath::abs(UB)*tolrel ) exitflag = 2;
    else if(LB > th) exitflag = 3;
    else if(t >= tmax) exitflag = 0; 

    if( verb && (t % verb) == 0) {
     SG_PRINT("%d: UB=%f,LB=%f,UB-LB=%f,(UB-LB)/|UB|=%f\n",
        t, UB, LB, UB-LB,(UB-LB)/UB); 
    }  

    /* Store selected values */
    if( t < History_size ) {
      History[INDEX(0,t,2)] = LB;
      History[INDEX(1,t,2)] = UB;
    }
    else {
      tmp_ptr = new DREAL[(History_size+HISTORY_BUF)*2];
      if( tmp_ptr == NULL ) SG_ERROR("Not enough memory.\n");
      for( i = 0; i < History_size; i++ ) {
        tmp_ptr[INDEX(0,i,2)] = History[INDEX(0,i,2)];
        tmp_ptr[INDEX(1,i,2)] = History[INDEX(1,i,2)];
      }
      tmp_ptr[INDEX(0,t,2)] = LB;
      tmp_ptr[INDEX(1,t,2)] = UB;
      
      History_size += HISTORY_BUF;
      delete[] History;
      History = tmp_ptr;
    }
  }

  /* print info about last iteration*/
  if(verb && (t % verb) ) {
    SG_PRINT("Exit: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f \n",
      UB, LB, UB-LB,(UB-LB)/UB);
  }  

  /*------------------------------------------------------- */
  /* Set outputs                                            */
  /*------------------------------------------------------- */
  (*ptr_t) = t;
  (*ptr_aHa11) = aHa11;
  (*ptr_aHa22) = aHa22;
  (*ptr_History) = History;

  /* Free memory */
  delete[] Ha1 ;
  delete[] Ha2;
  
  return( exitflag ); 
}


/* --------------------------------------------------------------
 QP solver based on Improved MDM algorithm (u fixed v optimized)

 Usage: exitflag = gnpp_imdm( diag_H, vector_c, vector_y,
       dim, tmax, tolabs, tolrel, th, &alpha, &t, &aHa11, &aHa22, &History );
-------------------------------------------------------------- */
int CGNPPLib::gnpp_imdm(double *diag_H,
            double *vector_c,
            double *vector_y,
            INT dim, 
            INT tmax,
            double tolabs,
            double tolrel,
            double th,
            double *alpha,
            INT  *ptr_t, 
            double *ptr_aHa11,
            double *ptr_aHa22,
            double **ptr_History,
            INT verb)
{
  double LB;
  double UB;
  double aHa11, aHa12, aHa22, ac1, ac2;
  double tmp;
  double Huu, Huv, Hvv;
  double min_beta1, max_beta1, min_beta2, max_beta2, beta;
  double lambda;
  double delta1, delta2;
  double improv, max_improv;
  double *History;
  double *Ha1;
  double *Ha2;
  double *tmp_ptr;
  double *col_u, *col_v;
  double *col_v1, *col_v2;
  long u1=0, u2=0;
  long v1, v2;
  long i;
  long t;
  long History_size;
  int exitflag;
  int which_case;

  /* ------------------------------------------------------------ */
  /* Initialization                                               */
  /* ------------------------------------------------------------ */

  Ha1 = new DREAL[dim];
  if( Ha1 == NULL ) SG_ERROR("Not enough memory.\n");
  Ha2 = new DREAL[dim];
  if( Ha2 == NULL ) SG_ERROR("Not enough memory.\n");

  History_size = (tmax < HISTORY_BUF ) ? tmax+1 : HISTORY_BUF;
  History = new DREAL[History_size*2];
  if( History == NULL ) SG_ERROR("Not enough memory.\n");

  /* inx1 = firts of find( y ==1 ), inx2 = firts of find( y ==2 ) */
  v1 = -1; v2 = -1; i = 0;
  while( (v1 == -1 || v2 == -1) && i < dim ) {
    if( v1 == -1 && vector_y[i] == 1 ) { v1 = i; }
    if( v2 == -1 && vector_y[i] == 2 ) { v2 = i; } 
    i++;
  }

  col_v1 = (double*)get_col(v1,-1);
  col_v2 = (double*)get_col(v2,v1);
  
  aHa12 = col_v1[v2];
  aHa11 = diag_H[v1];
  aHa22 = diag_H[v2];
  ac1 = vector_c[v1];
  ac2 = vector_c[v2];

  min_beta1 = PLUS_INF; min_beta2 = PLUS_INF;
  for( i = 0; i < dim; i++ ) 
  {
    alpha[i] = 0;
    Ha1[i] = col_v1[i];
    Ha2[i] = col_v2[i];

    beta = Ha1[i] + Ha2[i] + vector_c[i];

    if( vector_y[i] == 1 && min_beta1 > beta ) {
      u1 = i;
      min_beta1 = beta;
    }

    if( vector_y[i] == 2 && min_beta2 > beta ) {
      u2 = i;
      min_beta2 = beta;
    }
  }

  alpha[v1] = 1;
  alpha[v2] = 1;

  UB = 0.5*(aHa11 + 2*aHa12 + aHa22) + ac1 + ac2;
  LB = min_beta1 + min_beta2 - 0.5*(aHa11 + 2*aHa12 + aHa22);

  delta1 = Ha1[v1] + Ha2[v1] + vector_c[v1] - min_beta1;
  delta2 = Ha1[v2] + Ha2[v2] + vector_c[v2] - min_beta2;

  t = 0;
  History[INDEX(0,0,2)] = LB;
  History[INDEX(1,0,2)] = UB;

  if( verb ) {
    SG_PRINT("Init: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f \n",
      UB, LB, UB-LB,(UB-LB)/UB);
  }  

  if( delta1 > delta2 ) 
  {
     which_case = 1;
     col_u = (double*)get_col(u1,v1);
     col_v = col_v1;
  }
  else
  {
     which_case = 2;
     col_u = (double*)get_col(u2,v2);
     col_v = col_v2;
  }

  /* Stopping conditions */
  if( UB-LB <= tolabs ) exitflag = 1;
  else if(UB-LB <= CMath::abs(UB)*tolrel ) exitflag = 2;
  else if(LB > th) exitflag = 3;
  else exitflag = -1;

  /* ------------------------------------------------------------ */
  /* Main optimization loop                                       */
  /* ------------------------------------------------------------ */

  while( exitflag == -1 ) 
  {
    t++;     

    if( which_case == 1 )
    {
      Huu = diag_H[u1];
      Hvv = diag_H[v1];
      Huv = col_u[v1];

      lambda = delta1/(alpha[v1]*(Huu - 2*Huv + Hvv ));
      lambda = CMath::min(1.0,lambda);

      tmp = lambda*alpha[v1];

      aHa11 = aHa11 + 2*tmp*(Ha1[u1]-Ha1[v1])+tmp*tmp*( Huu - 2*Huv + Hvv );
      aHa12 = aHa12 + tmp*(Ha2[u1]-Ha2[v1]);
      ac1 = ac1 + tmp*(vector_c[u1]-vector_c[v1]);

      alpha[u1] = alpha[u1] + tmp;
      alpha[v1] = alpha[v1] - tmp;

      min_beta1 = PLUS_INF; min_beta2 = PLUS_INF;
      max_beta1 = MINUS_INF; max_beta2 = MINUS_INF; 
      for( i = 0; i < dim; i ++ )
      {
         Ha1[i] = Ha1[i] + tmp*(col_u[i] - col_v[i]);

         beta = Ha1[i] + Ha2[i] + vector_c[i];
         if( vector_y[i] == 1 ) 
           {
             if( min_beta1 > beta ) { u1 = i; min_beta1 = beta; }
             if( max_beta1 < beta && alpha[i] > 0 ) { v1 = i; max_beta1 = beta; }
           }
         else
           {
             if( min_beta2 > beta ) { u2 = i; min_beta2 = beta; }
             if( max_beta2 < beta && alpha[i] > 0) { v2 = i; max_beta2 = beta; }
           }
      }
    }
    else
    {
      Huu = diag_H[u2];
      Hvv = diag_H[v2];
      Huv = col_u[v2];
  
      lambda = delta2/(alpha[v2]*( Huu - 2*Huv + Hvv ));
      lambda = CMath::min(1.0,lambda);

      tmp = lambda*alpha[v2];
      aHa22 = aHa22 + 2*tmp*( Ha2[u2]-Ha2[v2]) + tmp*tmp*( Huu - 2*Huv + Hvv);
      aHa12 = aHa12 + tmp*(Ha1[u2]-Ha1[v2]);
      ac2 = ac2 + tmp*( vector_c[u2]-vector_c[v2] );

      alpha[u2] = alpha[u2] + tmp;
      alpha[v2] = alpha[v2] - tmp;

      min_beta1 = PLUS_INF; min_beta2 = PLUS_INF;
      max_beta1 = MINUS_INF; max_beta2 = MINUS_INF; 
      for(i = 0; i < dim; i++ ) 
      {  
         Ha2[i] = Ha2[i] + tmp*( col_u[i] - col_v[i] );

         beta = Ha1[i] + Ha2[i] + vector_c[i];

         if( vector_y[i] == 1 ) 
         {
           if( min_beta1 > beta ) { u1 = i; min_beta1 = beta; }
           if( max_beta1 < beta && alpha[i] > 0 ) { v1 = i; max_beta1 = beta; }
         }
         else
         {
           if( min_beta2 > beta ) { u2 = i; min_beta2 = beta; }
           if( max_beta2 < beta && alpha[i] > 0) { v2 = i; max_beta2 = beta; }
         }
      }
    }

    UB = 0.5*(aHa11 + 2*aHa12 + aHa22) + ac1 + ac2;
    LB = min_beta1 + min_beta2 - 0.5*(aHa11 + 2*aHa12 + aHa22);
  
    delta1 = Ha1[v1] + Ha2[v1] + vector_c[v1] - min_beta1;
    delta2 = Ha1[v2] + Ha2[v2] + vector_c[v2] - min_beta2;

    if( delta1 > delta2 ) 
    {
       col_u = (double*)get_col(u1,-1);

      /* search for optimal v while u is fixed */
      for( max_improv =  MINUS_INF, i = 0; i < dim; i++ ) {

        if( vector_y[i] == 1 && alpha[i] != 0 ) {

          beta = Ha1[i] + Ha2[i] + vector_c[i];

          if( beta >= min_beta1 ) {

            tmp = diag_H[u1] - 2*col_u[i] + diag_H[i];
            if( tmp != 0 ) {
              improv = (0.5*(beta-min_beta1)*(beta-min_beta1))/tmp;

              if( improv > max_improv ) {
                max_improv = improv;
                v1 = i;
              }
            }
          }
        }
      }
      col_v = (double*)get_col(v1,u1);
      delta1 = Ha1[v1] + Ha2[v1] + vector_c[v1] - min_beta1;
      which_case = 1;
      
    }
    else
    {
       col_u = (double*)get_col(u2,-1);

      /* search for optimal v while u is fixed */
      for( max_improv =  MINUS_INF, i = 0; i < dim; i++ ) {

        if( vector_y[i] == 2 && alpha[i] != 0 ) {

          beta = Ha1[i] + Ha2[i] + vector_c[i];

          if( beta >= min_beta2 ) {

            tmp = diag_H[u2] - 2*col_u[i] + diag_H[i];
            if( tmp != 0 ) {
              improv = (0.5*(beta-min_beta2)*(beta-min_beta2))/tmp;

              if( improv > max_improv ) {
                max_improv = improv;
                v2 = i;
              }
            }
          }
        }
      }

      col_v = (double*)get_col(v2,u2);
      delta2 = Ha1[v2] + Ha2[v2] + vector_c[v2] - min_beta2;
      which_case = 2;
    }
    

    /* Stopping conditions */
    if( UB-LB <= tolabs ) exitflag = 1; 
    else if( UB-LB <= CMath::abs(UB)*tolrel ) exitflag = 2;
    else if(LB > th) exitflag = 3;
    else if(t >= tmax) exitflag = 0; 

    if( verb && (t % verb) == 0) {
     SG_PRINT("%d: UB=%f,LB=%f,UB-LB=%f,(UB-LB)/|UB|=%f\n",
        t, UB, LB, UB-LB,(UB-LB)/UB); 
    }  

    /* Store selected values */
    if( t < History_size ) {
      History[INDEX(0,t,2)] = LB;
      History[INDEX(1,t,2)] = UB;
    }
    else {
      tmp_ptr = new DREAL[(History_size+HISTORY_BUF)*2];
      if( tmp_ptr == NULL ) SG_ERROR("Not enough memory.\n");
      for( i = 0; i < History_size; i++ ) {
        tmp_ptr[INDEX(0,i,2)] = History[INDEX(0,i,2)];
        tmp_ptr[INDEX(1,i,2)] = History[INDEX(1,i,2)];
      }
      tmp_ptr[INDEX(0,t,2)] = LB;
      tmp_ptr[INDEX(1,t,2)] = UB;
      
      History_size += HISTORY_BUF;
      delete[] History;
      History = tmp_ptr;
    }
  }

  /* print info about last iteration*/
  if(verb && (t % verb) ) {
    SG_PRINT("Exit: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f \n",
      UB, LB, UB-LB,(UB-LB)/UB);
  }  

  /*------------------------------------------------------- */
  /* Set outputs                                            */
  /*------------------------------------------------------- */
  (*ptr_t) = t;
  (*ptr_aHa11) = aHa11;
  (*ptr_aHa22) = aHa22;
  (*ptr_History) = History;

  /* Free memory */
  delete[] Ha1;
  delete[] Ha2;
  
  return( exitflag ); 
}


DREAL* CGNPPLib::get_col( long a, long b ) 
{
  double *col_ptr;
  double y;
  long i;
  long inx;

  inx = -1;
  for( i=0; i < Cache_Size; i++ ) {
    if( cache_index[i] == a ) { inx = i; break; }
  }
    
  if( inx != -1 ) {
    col_ptr = kernel_columns[inx];
    return( col_ptr );
  }
   
  col_ptr = kernel_columns[first_kernel_inx];
  cache_index[first_kernel_inx] = a;

  first_kernel_inx++;
  if( first_kernel_inx >= Cache_Size ) first_kernel_inx = 0;

  y = m_vector_y[a];
  for( i=0; i < m_num_data; i++ ) {
    if( m_vector_y[i] == y )  
    {
      col_ptr[i] = 2*m_kernel->kernel(i,a); 
    }
    else 
    {
      col_ptr[i] = -2*m_kernel->kernel(i,a);
    }
  }

  col_ptr[a] = col_ptr[a] + m_reg_const;

  return( col_ptr );
}



