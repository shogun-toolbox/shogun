/*-----------------------------------------------------------------------
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Library for solving QP task required for learning SVM without bias term.
 *
 * Written (W) 2006-2009 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 2006-2009 Center for Machine Perception, CTU FEL Prague
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST
 *
 *
 *  min 0.5*x'*H*x + f'*x
 *
 *  subject to  C >= x(i) >= 0 for all i
 *
 * H [dim x dim] is symmetric positive semi-definite matrix.
 * f [dim x 1] is an arbitrary vector.
 *
 * The precision of found solution is given by parameters
 * tmax, tolabs, tolrel which define the stopping conditions:
 *
 *    t >= tmax                   ->  exit_flag = 0  Number of iterations.
 *    UB-LB <= tolabs             ->  exit_flag = 1  Abs. tolerance.
 *    UB-LB <= UB*tolrel          ->  exit_flag = 2  Relative tolerance.
 *
 * UB ... Upper bound on the optimal solution.
 * LB ... Lower bound on the optimal solution.
 * t  ... Number of iterations.
 * History ... Value of LB and UB wrt. number of iterations.
 *
 * 1. Generalized Gauss-Seidel methods
 * exitflag = qpbsvm_sca( &get_col, diag_H, f, UB, dim, tmax,
 *               tolabs, tolrel, x, Nabla, &t, &History, verb )
 *
 * 2. Greedy variant - Udpate variable yielding the best improvement.
 * exitflag = qpbsvm_scas( &get_col, diag_H, f, UB, dim, tmax,
 *               tolabs, tolrel, x, Nabla, &t, &History, verb )
 *
 * 3. Updates variable which most violates the KKT conditions
 * exitflag = qpbsvm_scamv( &get_col, diag_H, f, UB, dim, tmax,
 *               tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb )
 *
-------------------------------------------------------------------- */

#include <shogun/mathematics/Math.h>
#include <string.h>
#include <limits.h>

#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Cplex.h>
#include <shogun/mathematics/Math.h>

#include <shogun/classifier/svm/QPBSVMLib.h>
#include <shogun/lib/external/pr_loqo.h>

using namespace shogun;

#define HISTORY_BUF 1000000

#define INDEX(ROW,COL,DIM) ((COL*DIM)+ROW)

CQPBSVMLib::CQPBSVMLib()
{
	SG_UNSTABLE("CQPBSVMLib::CQPBSVMLib()", "\n")

	m_H=0;
	m_dim = 0;
	m_diag_H = NULL;

	m_f = NULL;
	m_UB = 0.0;
	m_tmax = INT_MAX;
	m_tolabs = 0;
	m_tolrel = 1e-6;
	m_tolKKT = 0;
	m_solver = QPB_SOLVER_SCA;
}

CQPBSVMLib::CQPBSVMLib(
	float64_t* H, int32_t n, float64_t* f, int32_t m, float64_t UB)
: CSGObject()
{
	ASSERT(H && n>0)
	m_H=H;
	m_dim = n;
	m_diag_H=NULL;

	m_f=f;
	m_UB=UB;
	m_tmax = INT_MAX;
	m_tolabs = 0;
	m_tolrel = 1e-6;
	m_tolKKT = 0;
	m_solver = QPB_SOLVER_SCA;
}

CQPBSVMLib::~CQPBSVMLib()
{
	SG_FREE(m_diag_H);
}

int32_t CQPBSVMLib::solve_qp(float64_t* result, int32_t len)
{
	int32_t status = -1;
	ASSERT(len==m_dim)
	float64_t* Nabla=SG_MALLOC(float64_t, m_dim);
	for (int32_t i=0; i<m_dim; i++)
		Nabla[i]=m_f[i];

	SG_FREE(m_diag_H);
	m_diag_H=SG_MALLOC(float64_t, m_dim);

	for (int32_t i=0; i<m_dim; i++)
		m_diag_H[i]=m_H[i*m_dim+i];

	float64_t* History=NULL;
	int32_t t;
	int32_t verb=0;

	switch (m_solver)
	{
		case QPB_SOLVER_GRADDESC:
			status = qpbsvm_gradient_descent(result, Nabla, &t, &History, verb );
			break;
		case QPB_SOLVER_GS:
			status = qpbsvm_gauss_seidel(result, Nabla, &t, &History, verb );
			break;
		case QPB_SOLVER_SCA:
			status = qpbsvm_sca(result, Nabla, &t, &History, verb );
			break;
		case QPB_SOLVER_SCAS:
			status = qpbsvm_scas(result, Nabla, &t, &History, verb );
			break;
		case QPB_SOLVER_SCAMV:
			status = qpbsvm_scamv(result, Nabla, &t, &History, verb );
			break;
		case QPB_SOLVER_PRLOQO:
			status = qpbsvm_prloqo(result, Nabla, &t, &History, verb );
			break;
#ifdef USE_CPLEX
		case QPB_SOLVER_CPLEX:
			status = qpbsvm_cplex(result, Nabla, &t, &History, verb );
#else
			SG_ERROR("cplex not enabled at compile time - unknow solver\n")
#endif
			break;
		default:
			SG_ERROR("unknown solver\n")
			break;
	}

	SG_FREE(History);
	SG_FREE(Nabla);
	SG_FREE(m_diag_H);
	m_diag_H=NULL;

	return status;
}

/* --------------------------------------------------------------

Usage: exitflag = qpbsvm_sca(m_UB, m_dim, m_tmax,
               m_tolabs, m_tolrel, m_tolKKT, x, Nabla, &t, &History, verb )

-------------------------------------------------------------- */
int32_t CQPBSVMLib::qpbsvm_sca(float64_t *x,
	        float64_t *Nabla,
            int32_t   *ptr_t,
            float64_t **ptr_History,
            int32_t   verb)
{
  float64_t *History;
  float64_t *col_H;
  float64_t *tmp_ptr;
  float64_t x_old;
  float64_t delta_x;
  float64_t xHx;
  float64_t Q_P;
  float64_t Q_D;
  float64_t xf;
  float64_t xi_sum;
  int32_t History_size;
  int32_t t;
  int32_t i, j;
  int32_t exitflag;
  int32_t KKTsatisf;

  /* ------------------------------------------------------------ */
  /* Initialization                                               */
  /* ------------------------------------------------------------ */

  t = 0;

  History_size = (m_tmax < HISTORY_BUF ) ? m_tmax+1 : HISTORY_BUF;
  History=SG_MALLOC(float64_t, History_size*2);
  memset(History, 0, sizeof(float64_t)*History_size*2);

  /* compute Q_P and Q_D */
  xHx = 0;
  xf = 0;
  xi_sum = 0;
  for(i = 0; i < m_dim; i++ ) {
    xHx += x[i]*(Nabla[i] - m_f[i]);
    xf += x[i]*m_f[i];
    xi_sum += CMath::max(0.0,-Nabla[i]);
  }

  Q_P = 0.5*xHx + xf;
  Q_D = -0.5*xHx - m_UB*xi_sum;
  History[INDEX(0,t,2)] = Q_P;
  History[INDEX(1,t,2)] = Q_D;

  if( verb > 0 ) {
    SG_PRINT("%d: Q_P=%m_f, Q_D=%m_f, Q_P-Q_D=%m_f, (Q_P-Q_D)/|Q_P|=%m_f \n",
     t, Q_P, Q_D, Q_P-Q_D,(Q_P-Q_D)/CMath::abs(Q_P));
  }

  exitflag = -1;
  while( exitflag == -1 )
  {
    t++;

    for(i = 0; i < m_dim; i++ ) {
      if( m_diag_H[i] > 0 ) {
        /* variable update */
        x_old = x[i];
        x[i] = CMath::min(m_UB,CMath::max(0.0, x[i] - Nabla[i]/m_diag_H[i]));

        /* update Nabla */
        delta_x = x[i] - x_old;
        if( delta_x != 0 ) {
          col_H = (float64_t*)get_col(i);
          for(j = 0; j < m_dim; j++ ) {
            Nabla[j] += col_H[j]*delta_x;
          }
        }

      }
    }

    /* compute Q_P and Q_D */
    xHx = 0;
    xf = 0;
    xi_sum = 0;
    KKTsatisf = 1;
    for(i = 0; i < m_dim; i++ ) {
      xHx += x[i]*(Nabla[i] - m_f[i]);
      xf += x[i]*m_f[i];
      xi_sum += CMath::max(0.0,-Nabla[i]);

      if((x[i] > 0 && x[i] < m_UB && CMath::abs(Nabla[i]) > m_tolKKT) ||
         (x[i] == 0 && Nabla[i] < -m_tolKKT) ||
         (x[i] == m_UB && Nabla[i] > m_tolKKT)) KKTsatisf = 0;
    }

    Q_P = 0.5*xHx + xf;
    Q_D = -0.5*xHx - m_UB*xi_sum;

    /* stopping conditions */
    if(t >= m_tmax) exitflag = 0;
    else if(Q_P-Q_D <= m_tolabs) exitflag = 1;
    else if(Q_P-Q_D <= CMath::abs(Q_P)*m_tolrel) exitflag = 2;
    else if(KKTsatisf == 1) exitflag = 3;

    if( verb > 0 && (t % verb == 0 || t==1)) {
      SG_PRINT("%d: Q_P=%m_f, Q_D=%m_f, Q_P-Q_D=%m_f, (Q_P-Q_D)/|Q_P|=%m_f \n",
        t, Q_P, Q_D, Q_P-Q_D,(Q_P-Q_D)/CMath::abs(Q_P));
    }

    /* Store m_UB LB to History buffer */
    if( t < History_size ) {
      History[INDEX(0,t,2)] = Q_P;
      History[INDEX(1,t,2)] = Q_D;
    }
    else {
      tmp_ptr=SG_MALLOC(float64_t, (History_size+HISTORY_BUF)*2);
      memset(tmp_ptr, 0, sizeof(float64_t)*(History_size+HISTORY_BUF)*2);

      for( i = 0; i < History_size; i++ ) {
        tmp_ptr[INDEX(0,i,2)] = History[INDEX(0,i,2)];
        tmp_ptr[INDEX(1,i,2)] = History[INDEX(1,i,2)];
      }
      tmp_ptr[INDEX(0,t,2)] = Q_P;
      tmp_ptr[INDEX(1,t,2)] = Q_D;

      History_size += HISTORY_BUF;
      SG_FREE(History);
      History = tmp_ptr;
    }
  }

  (*ptr_t) = t;
  (*ptr_History) = History;

  SG_PRINT("QP: %f QD: %f\n", Q_P, Q_D)

  return( exitflag );
}


/* --------------------------------------------------------------

Usage: exitflag = qpbsvm_scas(m_UB, m_dim, m_tmax,
               m_tolabs, m_tolrel, m_tolKKT, x, Nabla, &t, &History, verb )

-------------------------------------------------------------- */
int32_t CQPBSVMLib::qpbsvm_scas(float64_t *x,
	        float64_t *Nabla,
            int32_t   *ptr_t,
            float64_t **ptr_History,
            int32_t   verb)
{
  float64_t *History;
  float64_t *col_H;
  float64_t *tmp_ptr;
  float64_t x_old;
  float64_t x_new;
  float64_t delta_x;
  float64_t max_x=CMath::INFTY;
  float64_t xHx;
  float64_t Q_P;
  float64_t Q_D;
  float64_t xf;
  float64_t xi_sum;
  float64_t max_update;
  float64_t curr_update;
  int32_t History_size;
  int32_t t;
  int32_t i, j;
  int32_t max_i=-1;
  int32_t exitflag;
  int32_t KKTsatisf;

  /* ------------------------------------------------------------ */
  /* Initialization                                               */
  /* ------------------------------------------------------------ */

  t = 0;

  History_size = (m_tmax < HISTORY_BUF ) ? m_tmax+1 : HISTORY_BUF;
  History=SG_MALLOC(float64_t, History_size*2);
  memset(History, 0, sizeof(float64_t)*History_size*2);

  /* compute Q_P and Q_D */
  xHx = 0;
  xf = 0;
  xi_sum = 0;
  for(i = 0; i < m_dim; i++ ) {
    xHx += x[i]*(Nabla[i] - m_f[i]);
    xf += x[i]*m_f[i];
    xi_sum += CMath::max(0.0,-Nabla[i]);
  }

  Q_P = 0.5*xHx + xf;
  Q_D = -0.5*xHx - m_UB*xi_sum;
  History[INDEX(0,t,2)] = Q_P;
  History[INDEX(1,t,2)] = Q_D;

  if( verb > 0 ) {
    SG_PRINT("%d: Q_P=%m_f, Q_D=%m_f, Q_P-Q_D=%m_f, (Q_P-Q_D)/|Q_P|=%m_f \n",
     t, Q_P, Q_D, Q_P-Q_D,(Q_P-Q_D)/CMath::abs(Q_P));
  }

  exitflag = -1;
  while( exitflag == -1 )
  {
    t++;

    max_update = -CMath::INFTY;
    for(i = 0; i < m_dim; i++ ) {
      if( m_diag_H[i] > 0 ) {
        /* variable update */
        x_old = x[i];
        x_new = CMath::min(m_UB,CMath::max(0.0, x[i] - Nabla[i]/m_diag_H[i]));

        curr_update = -0.5*m_diag_H[i]*(x_new*x_new-x_old*x_old) -
          (Nabla[i] - m_diag_H[i]*x_old)*(x_new - x_old);

        if( curr_update > max_update ) {
          max_i = i;
          max_update = curr_update;
          max_x = x_new;
        }
      }
    }

    x_old = x[max_i];
    x[max_i] = max_x;

    /* update Nabla */
    delta_x = max_x - x_old;
    if( delta_x != 0 ) {
      col_H = (float64_t*)get_col(max_i);
      for(j = 0; j < m_dim; j++ ) {
        Nabla[j] += col_H[j]*delta_x;
      }
    }

    /* compute Q_P and Q_D */
    xHx = 0;
    xf = 0;
    xi_sum = 0;
    KKTsatisf = 1;
    for(i = 0; i < m_dim; i++ ) {
      xHx += x[i]*(Nabla[i] - m_f[i]);
      xf += x[i]*m_f[i];
      xi_sum += CMath::max(0.0,-Nabla[i]);

      if((x[i] > 0 && x[i] < m_UB && CMath::abs(Nabla[i]) > m_tolKKT) ||
         (x[i] == 0 && Nabla[i] < -m_tolKKT) ||
         (x[i] == m_UB && Nabla[i] > m_tolKKT)) KKTsatisf = 0;
    }

    Q_P = 0.5*xHx + xf;
    Q_D = -0.5*xHx - m_UB*xi_sum;

    /* stopping conditions */
    if(t >= m_tmax) exitflag = 0;
    else if(Q_P-Q_D <= m_tolabs) exitflag = 1;
    else if(Q_P-Q_D <= CMath::abs(Q_P)*m_tolrel) exitflag = 2;
    else if(KKTsatisf == 1) exitflag = 3;

    if( verb > 0 && (t % verb == 0 || t==1)) {
      SG_PRINT("%d: Q_P=%m_f, Q_D=%m_f, Q_P-Q_D=%m_f, (Q_P-Q_D)/|Q_P|=%m_f \n",
        t, Q_P, Q_D, Q_P-Q_D,(Q_P-Q_D)/CMath::abs(Q_P));
    }

    /* Store m_UB LB to History buffer */
    if( t < History_size ) {
      History[INDEX(0,t,2)] = Q_P;
      History[INDEX(1,t,2)] = Q_D;
    }
    else {
      tmp_ptr=SG_MALLOC(float64_t, (History_size+HISTORY_BUF)*2);
      memset(tmp_ptr, 0, (History_size+HISTORY_BUF)*2*sizeof(float64_t));
      for( i = 0; i < History_size; i++ ) {
        tmp_ptr[INDEX(0,i,2)] = History[INDEX(0,i,2)];
        tmp_ptr[INDEX(1,i,2)] = History[INDEX(1,i,2)];
      }
      tmp_ptr[INDEX(0,t,2)] = Q_P;
      tmp_ptr[INDEX(1,t,2)] = Q_D;

      History_size += HISTORY_BUF;
      SG_FREE(History);
      History = tmp_ptr;
    }
  }

  (*ptr_t) = t;
  (*ptr_History) = History;

  return( exitflag );
}

/* --------------------------------------------------------------

Usage: exitflag = qpbsvm_scamv(m_UB, m_dim, m_tmax,
               m_tolabs, m_tolrel, m_tolKKT, x, Nabla, &t, &History, verb )

-------------------------------------------------------------- */
int32_t CQPBSVMLib::qpbsvm_scamv(float64_t *x,
	        float64_t *Nabla,
            int32_t   *ptr_t,
            float64_t **ptr_History,
            int32_t   verb)
{
  float64_t *History;
  float64_t *col_H;
  float64_t delta_x;
  float64_t x_new;
  float64_t max_viol;
  float64_t fval;
  int32_t t;
  int32_t i;
  int32_t u=-1;
  int32_t exitflag;

  /* ------------------------------------------------------------ */
  /* Initialization                                               */
  /* ------------------------------------------------------------ */

  t = 0;
  exitflag = -1;
  while( exitflag == -1 && t <= m_tmax)
  {
    t++;

    max_viol = 0;
    for(i = 0; i < m_dim; i++ )
    {
      if( x[i] == 0 )
      {
        if( max_viol < -Nabla[i]) { u = i; max_viol = -Nabla[i]; }
      }
      else if( x[i] > 0 && x[i] < m_UB )
      {
        if( max_viol < CMath::abs(Nabla[i]) ) { u = i; max_viol = CMath::abs(Nabla[i]); }
      }
      else if( max_viol < Nabla[i]) { u = i; max_viol = Nabla[i]; }
    }

/*    SG_PRINT("%d: max_viol=%m_f, u=%d\n", t, max_viol, u)*/

    if( max_viol <= m_tolKKT )
    {
      exitflag = 1;
    }
    else
    {
      /* update */
      x_new = CMath::min(m_UB,CMath::max(0.0, x[u] - Nabla[u]/m_diag_H[u]));

      delta_x = x_new - x[u];
      x[u] = x_new;

      col_H = (float64_t*)get_col(u);
      for(i = 0; i < m_dim; i++ ) {
        Nabla[i] += col_H[i]*delta_x;
      }
    }
  }

  History=SG_MALLOC(float64_t, (t+1)*2);
  memset(History, 0, sizeof(float64_t)*(t+1)*2);

  fval = 0;
  for(fval = 0, i = 0; i < m_dim; i++ ) {
    fval += 0.5*x[i]*(Nabla[i]+m_f[i]);
  }

  History[INDEX(0,t,2)] = fval;
  History[INDEX(1,t,2)] = 0;

  (*ptr_t) = t;
  (*ptr_History) = History;



  return( exitflag );
}

/* --------------------------------------------------------------

Usage: exitflag = qpbsvm_prloqo(m_UB, m_dim, m_tmax,
               m_tolabs, m_tolrel, m_tolKKT, x, Nabla, &t, &History, verb )

-------------------------------------------------------------- */
int32_t CQPBSVMLib::qpbsvm_prloqo(float64_t *x,
	        float64_t *Nabla,
            int32_t   *ptr_t,
            float64_t **ptr_History,
            int32_t   verb)
{
	float64_t* lb=SG_MALLOC(float64_t, m_dim);
	float64_t* ub=SG_MALLOC(float64_t, m_dim);
	float64_t* primal=SG_MALLOC(float64_t, 3*m_dim);
	float64_t* dual=SG_MALLOC(float64_t, 1+2*m_dim);
	float64_t* a=SG_MALLOC(float64_t, m_dim);

	for (int32_t i=0; i<m_dim; i++)
	{
		a[i]=0.0;
		lb[i]=0;
		ub[i]=m_UB;
	}

	float64_t b=0;

	SGVector<float64_t>::display_vector(m_f, m_dim, "m_f");
	int32_t result=pr_loqo(m_dim, 1, m_f, m_H, a, &b, lb, ub, primal, dual,
			2, 5, 1, -0.95, 10,0);

	SG_FREE(a);
	SG_FREE(lb);
	SG_FREE(ub);
	SG_FREE(primal);
	SG_FREE(dual);

	*ptr_t=0;
	*ptr_History=NULL;
	return result;
}

int32_t CQPBSVMLib::qpbsvm_gauss_seidel(float64_t *x,
	        float64_t *Nabla,
            int32_t   *ptr_t,
            float64_t **ptr_History,
            int32_t   verb)
{
	for (int32_t i=0; i<m_dim; i++)
		x[i]=CMath::random(0.0, 1.0);

	for (int32_t t=0; t<200; t++)
	{
		for (int32_t i=0; i<m_dim; i++)
		{
			x[i]= (-m_f[i]-(SGVector<float64_t>::dot(x,&m_H[m_dim*i], m_dim) -
						m_H[m_dim*i+i]*x[i]))/m_H[m_dim*i+i];
			x[i]=CMath::clamp(x[i], 0.0, 1.0);
		}
	}

	int32_t atbound=0;
	for (int32_t i=0; i<m_dim; i++)
	{
		if (x[i]==0.0 || x[i]==1.0)
			atbound++;
	}
	SG_PRINT("atbound:%d of %d (%2.2f%%)\n", atbound, m_dim, ((float64_t) 100.0*atbound)/m_dim)
	*ptr_t=0;
	*ptr_History=NULL;
	return 0;
}

int32_t CQPBSVMLib::qpbsvm_gradient_descent(float64_t *x,
	        float64_t *Nabla,
            int32_t   *ptr_t,
            float64_t **ptr_History,
            int32_t   verb)
{
	for (int32_t i=0; i<m_dim; i++)
		x[i]=CMath::random(0.0, 1.0);

	for (int32_t t=0; t<2000; t++)
	{
		for (int32_t i=0; i<m_dim; i++)
		{
			x[i]-=0.001*(SGVector<float64_t>::dot(x,&m_H[m_dim*i], m_dim)+m_f[i]);
			x[i]=CMath::clamp(x[i], 0.0, 1.0);
		}
	}

	int32_t atbound=0;
	for (int32_t i=0; i<m_dim; i++)
	{
		if (x[i]==0.0 || x[i]==1.0)
			atbound++;
	}
	SG_PRINT("atbound:%d of %d (%2.2f%%)\n", atbound, m_dim, ((float64_t) 100.0*atbound)/m_dim)
	*ptr_t=0;
	*ptr_History=NULL;
	return 0;
}

#ifdef USE_CPLEX
/* --------------------------------------------------------------

Usage: exitflag = qpbsvm_prloqo(m_UB, m_dim, m_tmax,
               m_tolabs, m_tolrel, m_tolKKT, x, Nabla, &t, &History, verb )

-------------------------------------------------------------- */
int32_t CQPBSVMLib::qpbsvm_cplex(float64_t *x,
	        float64_t *Nabla,
            int32_t   *ptr_t,
            float64_t **ptr_History,
            int32_t   verb)
{
	float64_t* lb=SG_MALLOC(float64_t, m_dim);
	float64_t* ub=SG_MALLOC(float64_t, m_dim);

	for (int32_t i=0; i<m_dim; i++)
	{
		lb[i]=0;
		ub[i]=m_UB;
	}

	CCplex cplex;
	cplex.init(E_QP);
	cplex.setup_lp(m_f, NULL, 0, m_dim, NULL, lb, ub);
	cplex.setup_qp(m_H, m_dim);
	cplex.optimize(x);
	cplex.cleanup();

	SG_FREE(lb);
	SG_FREE(ub);

	*ptr_t=0;
	*ptr_History=NULL;
	return 0;
}
#endif
