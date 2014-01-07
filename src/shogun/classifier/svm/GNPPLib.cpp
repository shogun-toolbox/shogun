/*-----------------------------------------------------------------------
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Library of solvers for Generalized Nearest Point Problem (GNPP).
 *
 * Written (W) 1999-2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2008 Center for Machine Perception, CTU FEL Prague 
 *
-------------------------------------------------------------------- */

#include <math.h>
#include <limits.h>
#include <lib/common.h>
#include <io/SGIO.h>
#include <mathematics/Math.h>

#include <classifier/svm/GNPPLib.h>
#include <kernel/Kernel.h>

using namespace shogun;

#define HISTORY_BUF 1000000

#define MINUS_INF INT_MIN
#define PLUS_INF  INT_MAX

#define INDEX(ROW,COL,DIM) ((COL*DIM)+ROW)

CGNPPLib::CGNPPLib()
{
	SG_UNSTABLE("CGNPPLib::CGNPPLib()", "\n")

	kernel_columns = NULL;
	cache_index = NULL;
	first_kernel_inx = 0;
	Cache_Size = 0;
	m_num_data = 0;
	m_reg_const = 0.0;
	m_vector_y = NULL;
	m_kernel = NULL;
}

CGNPPLib::CGNPPLib(
	float64_t* vector_y, CKernel* kernel, int32_t num_data, float64_t reg_const)
: CSGObject()
{
  m_reg_const = reg_const;
  m_num_data = num_data;
  m_vector_y = vector_y;
  m_kernel = kernel;

  Cache_Size = ((int64_t) kernel->get_cache_size())*1024*1024/(sizeof(float64_t)*num_data);
  Cache_Size = CMath::min(Cache_Size, (int64_t) num_data);

  SG_INFO("using %d kernel cache lines\n", Cache_Size)
  ASSERT(Cache_Size>=2)

  /* allocates memory for kernel cache */
  kernel_columns = SG_MALLOC(float64_t*, Cache_Size);
  cache_index = SG_MALLOC(float64_t, Cache_Size);

  for(int32_t i = 0; i < Cache_Size; i++ ) 
  {
    kernel_columns[i] = SG_MALLOC(float64_t, num_data);
    cache_index[i] = -2;
  }
  first_kernel_inx = 0;

}

CGNPPLib::~CGNPPLib()
{
  for(int32_t i = 0; i < Cache_Size; i++ ) 
      SG_FREE(kernel_columns[i]);

  SG_FREE(cache_index);
  SG_FREE(kernel_columns);
}

/* --------------------------------------------------------------
 QP solver based on Mitchell-Demyanov-Malozemov  algorithm.

 Usage: exitflag = gnpp_mdm( diag_H, vector_c, vector_y,
       dim, tmax, tolabs, tolrel, th, &alpha, &t, &aHa11, &aHa22, &History );
-------------------------------------------------------------- */
int8_t CGNPPLib::gnpp_mdm(float64_t *diag_H,
                       float64_t *vector_c,
                       float64_t *vector_y,
                       int32_t dim,
                       int32_t tmax,
                       float64_t tolabs,
                       float64_t tolrel,
                       float64_t th,
                       float64_t *alpha,
                       int32_t  *ptr_t,
                       float64_t *ptr_aHa11,
                       float64_t *ptr_aHa22,
                       float64_t **ptr_History,
                       int32_t verb)
{
  float64_t LB;
  float64_t UB;
  float64_t aHa11, aHa12, aHa22, ac1, ac2;
  float64_t tmp;
  float64_t Huu, Huv, Hvv;
  float64_t min_beta1, max_beta1, min_beta2, max_beta2, beta;
  float64_t lambda;
  float64_t delta1, delta2;
  float64_t *History;
  float64_t *Ha1;
  float64_t *Ha2;
  float64_t *tmp_ptr;
  float64_t *col_u, *col_v;
  float64_t *col_v1, *col_v2;
  int64_t u1=0, u2=0;
  int64_t v1, v2;
  int64_t i;
  int64_t t;
  int64_t History_size;
  int8_t exitflag;

  /* ------------------------------------------------------------ */
  /* Initialization                                               */
  /* ------------------------------------------------------------ */

  Ha1 = SG_MALLOC(float64_t, dim);
  if( Ha1 == NULL ) SG_ERROR("Not enough memory.\n")
  Ha2 = SG_MALLOC(float64_t, dim);
  if( Ha2 == NULL ) SG_ERROR("Not enough memory.\n")

  History_size = (tmax < HISTORY_BUF ) ? tmax+1 : HISTORY_BUF;
  History = SG_MALLOC(float64_t, History_size*2);
  if( History == NULL ) SG_ERROR("Not enough memory.\n")

  /* inx1 = firts of find( y ==1 ), inx2 = firts of find( y ==2 ) */
  v1 = -1; v2 = -1; i = 0;
  while( (v1 == -1 || v2 == -1) && i < dim ) {
    if( v1 == -1 && vector_y[i] == 1 ) { v1 = i; }
    if( v2 == -1 && vector_y[i] == 2 ) { v2 = i; } 
    i++;
  }

  col_v1 = (float64_t*)get_col(v1,-1);
  col_v2 = (float64_t*)get_col(v2,v1);
  
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
      col_u = (float64_t*)get_col(u1,-1);
      col_v = (float64_t*)get_col(v1,u1);

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
      col_u = (float64_t*)get_col(u2,-1);
      col_v = (float64_t*)get_col(v2,u2);

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
      tmp_ptr = SG_MALLOC(float64_t, (History_size+HISTORY_BUF)*2);
      if( tmp_ptr == NULL ) SG_ERROR("Not enough memory.\n")
      for( i = 0; i < History_size; i++ ) {
        tmp_ptr[INDEX(0,i,2)] = History[INDEX(0,i,2)];
        tmp_ptr[INDEX(1,i,2)] = History[INDEX(1,i,2)];
      }
      tmp_ptr[INDEX(0,t,2)] = LB;
      tmp_ptr[INDEX(1,t,2)] = UB;
      
      History_size += HISTORY_BUF;
      SG_FREE(History);
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
  SG_FREE(Ha1);
  SG_FREE(Ha2);
  
  return( exitflag ); 
}


/* --------------------------------------------------------------
 QP solver based on Improved MDM algorithm (u fixed v optimized)

 Usage: exitflag = gnpp_imdm( diag_H, vector_c, vector_y,
       dim, tmax, tolabs, tolrel, th, &alpha, &t, &aHa11, &aHa22, &History );
-------------------------------------------------------------- */
int8_t CGNPPLib::gnpp_imdm(float64_t *diag_H,
            float64_t *vector_c,
            float64_t *vector_y,
            int32_t dim,
            int32_t tmax,
            float64_t tolabs,
            float64_t tolrel,
            float64_t th,
            float64_t *alpha,
            int32_t  *ptr_t,
            float64_t *ptr_aHa11,
            float64_t *ptr_aHa22,
            float64_t **ptr_History,
            int32_t verb)
{
  float64_t LB;
  float64_t UB;
  float64_t aHa11, aHa12, aHa22, ac1, ac2;
  float64_t tmp;
  float64_t Huu, Huv, Hvv;
  float64_t min_beta1, max_beta1, min_beta2, max_beta2, beta;
  float64_t lambda;
  float64_t delta1, delta2;
  float64_t improv, max_improv;
  float64_t *History;
  float64_t *Ha1;
  float64_t *Ha2;
  float64_t *tmp_ptr;
  float64_t *col_u, *col_v;
  float64_t *col_v1, *col_v2;
  int64_t u1=0, u2=0;
  int64_t v1, v2;
  int64_t i;
  int64_t t;
  int64_t History_size;
  int8_t exitflag;
  int8_t which_case;

  /* ------------------------------------------------------------ */
  /* Initialization                                               */
  /* ------------------------------------------------------------ */

  Ha1 = SG_MALLOC(float64_t, dim);
  if( Ha1 == NULL ) SG_ERROR("Not enough memory.\n")
  Ha2 = SG_MALLOC(float64_t, dim);
  if( Ha2 == NULL ) SG_ERROR("Not enough memory.\n")

  History_size = (tmax < HISTORY_BUF ) ? tmax+1 : HISTORY_BUF;
  History = SG_MALLOC(float64_t, History_size*2);
  if( History == NULL ) SG_ERROR("Not enough memory.\n")

  /* inx1 = firts of find( y ==1 ), inx2 = firts of find( y ==2 ) */
  v1 = -1; v2 = -1; i = 0;
  while( (v1 == -1 || v2 == -1) && i < dim ) {
    if( v1 == -1 && vector_y[i] == 1 ) { v1 = i; }
    if( v2 == -1 && vector_y[i] == 2 ) { v2 = i; } 
    i++;
  }

  col_v1 = (float64_t*)get_col(v1,-1);
  col_v2 = (float64_t*)get_col(v2,v1);
  
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
     col_u = (float64_t*)get_col(u1,v1);
     col_v = col_v1;
  }
  else
  {
     which_case = 2;
     col_u = (float64_t*)get_col(u2,v2);
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
       col_u = (float64_t*)get_col(u1,-1);

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
      col_v = (float64_t*)get_col(v1,u1);
      delta1 = Ha1[v1] + Ha2[v1] + vector_c[v1] - min_beta1;
      which_case = 1;
      
    }
    else
    {
       col_u = (float64_t*)get_col(u2,-1);

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

      col_v = (float64_t*)get_col(v2,u2);
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
      tmp_ptr = SG_MALLOC(float64_t, (History_size+HISTORY_BUF)*2);
      if( tmp_ptr == NULL ) SG_ERROR("Not enough memory.\n")
      for( i = 0; i < History_size; i++ ) {
        tmp_ptr[INDEX(0,i,2)] = History[INDEX(0,i,2)];
        tmp_ptr[INDEX(1,i,2)] = History[INDEX(1,i,2)];
      }
      tmp_ptr[INDEX(0,t,2)] = LB;
      tmp_ptr[INDEX(1,t,2)] = UB;
      
      History_size += HISTORY_BUF;
      SG_FREE(History);
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
  SG_FREE(Ha1);
  SG_FREE(Ha2);
  
  return( exitflag ); 
}


float64_t* CGNPPLib::get_col(int64_t a, int64_t b)
{
  float64_t *col_ptr;
  float64_t y;
  int64_t i;
  int64_t inx;

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
