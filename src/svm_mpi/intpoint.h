/* -*-C++-*- */

#ifndef __INTPOINT_H_
#define __INTPOINT_H_

#include "matrix.h"
#include "optimizer.h"
#if defined(HAVE_MPI) && !defined(DISABLE_MPI)
#include "intpoint_mpi.h"
#endif

#define PARALLEL_ONLY

template <class T>
void optimize(COptimizer &optimizer, CMatrix<T> &c, CMatrix<T> &H,
	      CMatrix<T> &A, CMatrix<T> &b, CMatrix<T> &l, 
	      CMatrix<T> &u, const CMatrix<T> &r, CMatrix<T> &primal, 
	      CMatrix<T> &dual, const char **how);

template <class T>
void optimize_smw(COptimizer &optimizer, CMatrix<T> &c, 
		  CMatrix<T> &Hmn, CMatrix<T> &Hnn,
		  CMatrix<T> &A, CMatrix<T> &b, CMatrix<T> &l, 
		  CMatrix<T> &u, const CMatrix<T> &r, CMatrix<T> &primal, 
		  CMatrix<T> &dual, const char **how);

template <class T>
void optimize_smw_linear(COptimizer &optimizer, CMatrix<T> &c,
			 CMatrix<T> &zmn, CMatrix<T> &A, CMatrix<T> &b,
			 CMatrix<T> &r, CMatrix<T> &l, CMatrix<T> &u,
			 CMatrix<T> &primal, CMatrix<T> &dual,
			 const char **how);


#ifdef HAVE_MPI
/* For root node */

template <class T>
void optimize_smw2mpi_core(COptimizer &optimizer, CMatrix<T> &c,
			   CMatrix<T> &p_kmnZ,
#ifndef PARALLEL_ONLY
			   CMatrix<T> &Z,
#endif
			   CMatrix<T> &A, CMatrix<T> &b,
			   CMatrix<T> &l, CMatrix<T> &u, const CMatrix<T> &r,
			   const unsigned m_prime, const unsigned m_last,
			   const unsigned my_rank, const unsigned num_nodes,
			   IntpointResources *resources, CMatrix<T> &primal,
			   CMatrix<T> &dual, const char **how);


void send_z_columns_double(MPI_Comm comm, double *data, const unsigned start_col,
			   const unsigned ncols, const unsigned nrows,
			   const int rank, const bool ishomogeneous) ;

#endif /* HAVE_MPI */

#endif /* ! __INTPOINT_H_ */
