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
 *
gmnplib.c: Library of solvers for Generalized Minimal Norm Problem (GMNP).

 Generalized Minimal Norm Problem to solve is

  min 0.5*alpha'*H*alpha + c'*alpha

  subject to  sum(alpha) = 1,  alpha(i) >= 0

 H [dim x dim] is symmetric positive definite matrix.
 c [dim x 1] is an arbitrary vector.

 The precision of the found solution is given by
 the parameters tmax, tolabs and tolrel which
 define the stopping conditions:

 UB-LB <= tolabs      ->  exit_flag = 1   Abs. tolerance.
 UB-LB <= UB*tolrel   ->  exit_flag = 2   Relative tolerance.
 LB > th              ->  exit_flag = 3   Threshold on lower bound.
 t >= tmax            ->  exit_flag = 0   Number of iterations.

 UB ... Upper bound on the optimal solution.
 LB ... Lower bound on the optimal solution.
 t  ... Number of iterations.
 History ... Value of LB and UB wrt. number of iterations.


 The following algorithms are implemented:
 ..............................................

 - GMNP solver based on improved MDM algorithm 1 (u fixed v optimized)
    exitflag = gmnp_imdm( &get_col, diag_H, vector_c, dim,
                 tmax, tolabs, tolrel, th, &alpha, &t, &History, verb  );

  For more info refer to V.Franc: Optimization Algorithms for Kernel
  Methods. Research report. CTU-CMP-2005-22. CTU FEL Prague. 2005.
  ftp://cmp.felk.cvut.cz/pub/cmp/articles/franc/Franc-PhD.pdf .

 Modifications:
 09-sep-2005, VF
 24-jan-2005, VF
 26-nov-2004, VF
 25-nov-2004, VF
 21-nov-2004, VF
 20-nov-2004, VF
 31-may-2004, VF
 23-Jan-2004, VF

-------------------------------------------------------------------- */

#include <multiclass/GMNPLib.h>
#include <mathematics/Math.h>

#include <string.h>
#include <limits.h>

using namespace shogun;

#define HISTORY_BUF 1000000

#define MINUS_INF INT_MIN
#define PLUS_INF  INT_MAX

#define INDEX(ROW,COL,DIM) ((COL*DIM)+ROW)
#define KDELTA(A,B) (A==B)
#define KDELTA4(A1,A2,A3,A4) ((A1==A2)||(A1==A3)||(A1==A4)||(A2==A3)||(A2==A4)||(A3==A4))

CGMNPLib::CGMNPLib()
{
	SG_UNSTABLE("CGMNPLib::CGMNPLib()", "\n")

	diag_H = NULL;
	kernel_columns = NULL;
	cache_index = NULL;
	first_kernel_inx = 0;
	Cache_Size = 0;
	m_num_data = 0;
	m_reg_const = 0;
	m_vector_y = 0;
	m_kernel = NULL;

	first_virt_inx = 0;
	memset(&virt_columns, 0, sizeof (virt_columns));
	m_num_virt_data = 0;
	m_num_classes = 0;
}

CGMNPLib::CGMNPLib(
	float64_t* vector_y, CKernel* kernel, int32_t num_data,
	int32_t num_virt_data, int32_t num_classes, float64_t reg_const)
: CSGObject()
{
  m_num_classes=num_classes;
  m_num_virt_data=num_virt_data;
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



  for(int32_t i = 0; i < 3; i++ )
  {
    virt_columns[i] = SG_MALLOC(float64_t, num_virt_data);
  }
  first_virt_inx = 0;

  diag_H = SG_MALLOC(float64_t, num_virt_data);

  for(int32_t i = 0; i < num_virt_data; i++ )
	  diag_H[i] = kernel_fce(i,i);
}

CGMNPLib::~CGMNPLib()
{
	for(int32_t i = 0; i < Cache_Size; i++ )
		SG_FREE(kernel_columns[i]);

	for(int32_t i = 0; i < 3; i++ )
		SG_FREE(virt_columns[i]);

	SG_FREE(cache_index);
	SG_FREE(kernel_columns);

	SG_FREE(diag_H);
}

/* ------------------------------------------------------------
  Returns pointer at a-th column of the kernel matrix.
  This function maintains FIFO cache of kernel columns.
------------------------------------------------------------ */
float64_t* CGMNPLib::get_kernel_col( int32_t a )
{
  float64_t *col_ptr;
  int32_t i;
  int32_t inx;

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

  for( i=0; i < m_num_data; i++ ) {
    col_ptr[i] = m_kernel->kernel(i,a);
  }

  return( col_ptr );
}

/* ------------------------------------------------------------
  Computes index of input example and its class label from
  index of virtual "single-class" example.
------------------------------------------------------------ */
void CGMNPLib::get_indices2( int32_t *index, int32_t *c, int32_t i )
{
   *index = i / (m_num_classes-1);

   *c= (i % (m_num_classes-1))+1;
   if( *c>= m_vector_y[ *index ]) (*c)++;

   return;
}


/* ------------------------------------------------------------
  Returns pointer at the a-th column of the virtual K matrix.

  (note: the b-th column must be preserved in the cache during
   updating but b is from (a(t-2), a(t-1)) where a=a(t) and
   thus FIFO with three columns does not have to take care od b.)
------------------------------------------------------------ */
float64_t* CGMNPLib::get_col( int32_t a, int32_t b )
{
  int32_t i;
  float64_t *col_ptr;
  float64_t *ker_ptr;
  float64_t value;
  int32_t i1,c1,i2,c2;

  col_ptr = virt_columns[first_virt_inx++];
  if( first_virt_inx >= 3 ) first_virt_inx = 0;

  get_indices2( &i1, &c1, a );
  ker_ptr = (float64_t*) get_kernel_col( i1 );

  for( i=0; i < m_num_virt_data; i++ ) {
    get_indices2( &i2, &c2, i );

    if( KDELTA4(m_vector_y[i1],m_vector_y[i2],c1,c2) ) {
      value = (+KDELTA(m_vector_y[i1],m_vector_y[i2])
               -KDELTA(m_vector_y[i1],c2)
               -KDELTA(m_vector_y[i2],c1)
               +KDELTA(c1,c2)
              )*(ker_ptr[i2]+1);
    }
    else
    {
      value = 0;
    }

    if(a==i) value += m_reg_const;

    col_ptr[i] = value;
  }

  return( col_ptr );
}


/* --------------------------------------------------------------
 GMNP solver based on improved MDM algorithm 1.

 Search strategy: u determined by common rule and v is
 optimized.

 Usage: exitflag = gmnp_imdm( &get_col, diag_H, vector_c, dim,
                  tmax, tolabs, tolrel, th, &alpha, &t, &History );
-------------------------------------------------------------- */

int8_t CGMNPLib::gmnp_imdm(float64_t *vector_c,
            int32_t dim,
            int32_t tmax,
            float64_t tolabs,
            float64_t tolrel,
            float64_t th,
            float64_t *alpha,
            int32_t  *ptr_t,
            float64_t **ptr_History,
            int32_t verb)
{
  float64_t LB;
  float64_t UB;
  float64_t aHa, ac;
  float64_t tmp, tmp1;
  float64_t Huu, Huv, Hvv;
  float64_t min_beta, beta;
  float64_t max_improv, improv;
  float64_t lambda;
  float64_t *History;
  float64_t *Ha;
  float64_t *tmp_ptr;
  float64_t *col_u, *col_v;
  int32_t u=0;
  int32_t v=0;
  int32_t new_u=0;
  int32_t i;
  int32_t t;
  int32_t History_size;
  int8_t exitflag;

  /* ------------------------------------------------------------ */
  /* Initialization                                               */
  /* ------------------------------------------------------------ */

  Ha = SG_MALLOC(float64_t, dim);
  if( Ha == NULL ) SG_ERROR("Not enough memory.")

  History_size = (tmax < HISTORY_BUF ) ? tmax+1 : HISTORY_BUF;
  History = SG_MALLOC(float64_t, History_size*2);
  if( History == NULL ) SG_ERROR("Not enough memory.")

  /* inx = argmin(0.5*diag_H + vector_c ); */
  for( tmp1 =  PLUS_INF, i = 0; i < dim; i++ ) {
    tmp = 0.5*diag_H[i] + vector_c[i];
    if( tmp1 > tmp) {
      tmp1 = tmp;
      v = i;
    }
  }

  col_v = (float64_t*)get_col(v,-1);

  for( min_beta = PLUS_INF, i = 0; i < dim; i++ )
  {
    alpha[i] = 0;
    Ha[i] = col_v[i];

    beta = Ha[i] + vector_c[i];
    if( beta < min_beta ) {
      min_beta = beta;
      u = i;
    }
  }

  alpha[v] = 1;
  aHa = diag_H[v];
  ac = vector_c[v];

  UB = 0.5*aHa + ac;
  LB = min_beta - 0.5*aHa;

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
  else if(LB > th ) exitflag = 3;
  else exitflag = -1;

  /* ------------------------------------------------------------ */
  /* Main optimization loop                                       */
  /* ------------------------------------------------------------ */

  col_u = (float64_t*)get_col(u,-1);
  while( exitflag == -1 )
  {
    t++;

    col_v = (float64_t*)get_col(v,u);

    /* Adaptation rule and update */
    Huu = diag_H[u];
    Hvv = diag_H[v];
    Huv = col_u[v];

    lambda = (Ha[v]-Ha[u]+vector_c[v]-vector_c[u])/(alpha[v]*(Huu-2*Huv+Hvv));
    if( lambda < 0 ) lambda = 0; else if (lambda > 1) lambda = 1;

    aHa = aHa + 2*alpha[v]*lambda*(Ha[u]-Ha[v])+
                lambda*lambda*alpha[v]*alpha[v]*(Huu-2*Huv+Hvv);

    ac = ac + lambda*alpha[v]*(vector_c[u]-vector_c[v]);

    tmp = alpha[v];
    alpha[u]=alpha[u]+lambda*alpha[v];
    alpha[v]=alpha[v]-lambda*alpha[v];

    UB = 0.5*aHa + ac;

/*    max_beta = MINUS_INF;*/
    for( min_beta = PLUS_INF, i = 0; i < dim; i++ )
    {
       Ha[i] = Ha[i] + lambda*tmp*(col_u[i] - col_v[i]);

       beta = Ha[i]+ vector_c[i];

       if( beta < min_beta )
       {
         new_u = i;
         min_beta = beta;
       }
    }

    LB = min_beta - 0.5*aHa;
    u = new_u;
    col_u = (float64_t*)get_col(u,-1);

    /* search for optimal v while u is fixed */
    for( max_improv =  MINUS_INF, i = 0; i < dim; i++ ) {

      if( alpha[i] != 0 ) {
        beta = Ha[i] + vector_c[i];

        if( beta >= min_beta ) {

          tmp = diag_H[u] - 2*col_u[i] + diag_H[i];
          if( tmp != 0 ) {
            improv = (0.5*(beta-min_beta)*(beta-min_beta))/tmp;

            if( improv > max_improv ) {
              max_improv = improv;
              v = i;
            }
          }
        }
      }
    }

    /* Stopping conditions */
    if( UB-LB <= tolabs ) exitflag = 1;
    else if( UB-LB <= CMath::abs(UB)*tolrel) exitflag = 2;
    else if(LB > th ) exitflag = 3;
    else if(t >= tmax) exitflag = 0;

    /* print info */
	SG_ABS_PROGRESS(CMath::abs((UB-LB)/UB),
			-CMath::log10(CMath::abs(UB-LB)),
			-CMath::log10(1.0),
			-CMath::log10(tolrel), 6);
    if(verb && (t % verb) == 0 ) {
      SG_PRINT("%d: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f \n",
        t, UB, LB, UB-LB,(UB-LB)/UB);
    }

    /* Store selected values */
    if( t < History_size ) {
      History[INDEX(0,t,2)] = LB;
      History[INDEX(1,t,2)] = UB;
    }
    else {
      tmp_ptr = SG_MALLOC(float64_t, (History_size+HISTORY_BUF)*2);
      if( tmp_ptr == NULL ) SG_ERROR("Not enough memory.")
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
  SG_DONE()
  if(verb && (t % verb) ) {
    SG_PRINT("exit: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f \n",
      UB, LB, UB-LB,(UB-LB)/UB);
  }


  /*------------------------------------------------------- */
  /* Set outputs                                            */
  /*------------------------------------------------------- */
  (*ptr_t) = t;
  (*ptr_History) = History;

  /* Free memory */
  SG_FREE(Ha);

  return( exitflag );
}

/* ------------------------------------------------------------
  Retures (a,b)-th element of the virtual kernel matrix
  of size [num_virt_data x num_virt_data].
------------------------------------------------------------ */
float64_t CGMNPLib::kernel_fce( int32_t a, int32_t b )
{
  float64_t value;
  int32_t i1,c1,i2,c2;

  get_indices2( &i1, &c1, a );
  get_indices2( &i2, &c2, b );

  if( KDELTA4(m_vector_y[i1],m_vector_y[i2],c1,c2) ) {
    value = (+KDELTA(m_vector_y[i1],m_vector_y[i2])
             -KDELTA(m_vector_y[i1],c2)
             -KDELTA(m_vector_y[i2],c1)
             +KDELTA(c1,c2)
            )*(m_kernel->kernel( i1, i2 )+1);
  }
  else
  {
    value = 0;
  }

  if(a==b) value += m_reg_const;

  return( value );
}
