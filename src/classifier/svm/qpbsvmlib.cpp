/*-----------------------------------------------------------------------
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Library for solving QP task required for learning SVM without bias term. 
 *
 * Written (W) 2006-2007 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 2006-2007 Center for Machine Perception, CTU FEL Prague 
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

#include <math.h>
#include <string.h>
#include <limits.h>

#include "classifier/svm/qpbsvmlib.h"
#include "lib/io.h"
#include "lib/Mathematics.h"

#define HISTORY_BUF 1000000

#define INDEX(ROW,COL,DIM) ((COL*DIM)+ROW)

CQPBSVMLib::CQPBSVMLib(DREAL* H, INT n, DREAL* f, INT m, DREAL UB)
{
	ASSERT(H && n>0);
	m_H=H;
	m_dim = n;

	m_diag_H=new DREAL[n];
	ASSERT(m_diag_H);

	for (INT i=0; i<n; i++)
		m_diag_H[i]=m_H[i*n+i];

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
	delete[] m_diag_H;
}

INT CQPBSVMLib::solve_qp(DREAL* result, INT len)
{
	ASSERT(len==m_dim);
	DREAL* Nabla=NULL;
	DREAL* History;
	INT t;
	INT verb=0;

	switch (m_solver)
	{
		case QPB_SOLVER_SCA:
			return qpbsvm_sca(result, Nabla, &t, &History, verb );
		case QPB_SOLVER_SCAS:
			return qpbsvm_scas(result, Nabla, &t, &History, verb );
		case QPB_SOLVER_SCAMV:
			return qpbsvm_scamv(result, Nabla, &t, &History, verb );
		default:
			SG_ERROR("unknown solver\n");
			return -1;
	}
}

/* --------------------------------------------------------------

Usage: exitflag = qpbsvm_sca(m_UB, m_dim, m_tmax, 
               m_tolabs, m_tolrel, m_tolKKT, x, Nabla, &t, &History, verb )

-------------------------------------------------------------- */
INT CQPBSVMLib::qpbsvm_sca(DREAL *x,
	        DREAL *Nabla,
            INT   *ptr_t,
            DREAL **ptr_History,
            INT   verb)
{
  DREAL *History;
  DREAL *col_H;
  DREAL *tmp_ptr;
  DREAL x_old;
  DREAL delta_x;
  DREAL xHx;
  DREAL Q_P;
  DREAL Q_D;
  DREAL xf;
  DREAL xi_sum;
  INT History_size;
  INT t;
  INT i, j;
  INT exitflag;
  INT KKTsatisf;

  /* ------------------------------------------------------------ */
  /* Initialization                                               */
  /* ------------------------------------------------------------ */

  t = 0;

  History_size = (m_tmax < HISTORY_BUF ) ? m_tmax+1 : HISTORY_BUF;
  History = new DREAL[History_size*2];
  ASSERT(History);
  memset(History, 0, sizeof(DREAL)*History_size*2);

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
          col_H = (DREAL*)get_col(i);
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
      tmp_ptr = new DREAL[(History_size+HISTORY_BUF)*2];
      ASSERT(tmp_ptr);
	  memset(tmp_ptr, 0, sizeof(DREAL)*(History_size+HISTORY_BUF)*2);

      for( i = 0; i < History_size; i++ ) {
        tmp_ptr[INDEX(0,i,2)] = History[INDEX(0,i,2)];
        tmp_ptr[INDEX(1,i,2)] = History[INDEX(1,i,2)];
      }
      tmp_ptr[INDEX(0,t,2)] = Q_P;
      tmp_ptr[INDEX(1,t,2)] = Q_D;
      
      History_size += HISTORY_BUF;
      delete[] History;
      History = tmp_ptr;
    }
  }

  (*ptr_t) = t;
  (*ptr_History) = History;

  return( exitflag ); 
}


/* --------------------------------------------------------------

Usage: exitflag = qpbsvm_scas(m_UB, m_dim, m_tmax, 
               m_tolabs, m_tolrel, m_tolKKT, x, Nabla, &t, &History, verb )

-------------------------------------------------------------- */
INT CQPBSVMLib::qpbsvm_scas(DREAL *x,
	        DREAL *Nabla,
            INT   *ptr_t,
            DREAL **ptr_History,
            INT   verb)
{
  DREAL *History;
  DREAL *col_H;
  DREAL *tmp_ptr;
  DREAL x_old;
  DREAL x_new;
  DREAL delta_x;
  DREAL max_x=CMath::INFTY;
  DREAL xHx;
  DREAL Q_P;
  DREAL Q_D;
  DREAL xf;
  DREAL xi_sum;
  DREAL max_update;
  DREAL curr_update;
  INT History_size;
  INT t;
  INT i, j;
  INT max_i=-1;
  INT exitflag;
  INT KKTsatisf;

  /* ------------------------------------------------------------ */
  /* Initialization                                               */
  /* ------------------------------------------------------------ */

  t = 0;

  History_size = (m_tmax < HISTORY_BUF ) ? m_tmax+1 : HISTORY_BUF;
  History = new DREAL[History_size*2];
  ASSERT(History);
  memset(History, 0, sizeof(DREAL)*History_size*2);

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
      col_H = (DREAL*)get_col(max_i);
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
      tmp_ptr = new DREAL[(History_size+HISTORY_BUF)*2];
      ASSERT(tmp_ptr);
	  memset(tmp_ptr, 0, (History_size+HISTORY_BUF)*2*sizeof(DREAL));
      for( i = 0; i < History_size; i++ ) {
        tmp_ptr[INDEX(0,i,2)] = History[INDEX(0,i,2)];
        tmp_ptr[INDEX(1,i,2)] = History[INDEX(1,i,2)];
      }
      tmp_ptr[INDEX(0,t,2)] = Q_P;
      tmp_ptr[INDEX(1,t,2)] = Q_D;
      
      History_size += HISTORY_BUF;
      delete[] History;
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
INT CQPBSVMLib::qpbsvm_scamv(DREAL *x,
	        DREAL *Nabla,
            INT   *ptr_t,
            DREAL **ptr_History,
            INT   verb)
{
  DREAL *History;
  DREAL *col_H;
  DREAL delta_x;
  DREAL x_new;
  DREAL max_viol;
  DREAL fval;
  INT t;
  INT i;
  INT u=-1;
  INT exitflag;

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

/*    SG_PRINT("%d: max_viol=%m_f, u=%d\n", t, max_viol, u);*/

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

      col_H = (DREAL*)get_col(u);
      for(i = 0; i < m_dim; i++ ) {
        Nabla[i] += col_H[i]*delta_x;
      }
    }
  }

  History = new DREAL[(t+1)*2];
  ASSERT(History);
  memset(History, 0, sizeof(DREAL)*(t+1)*2);

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
