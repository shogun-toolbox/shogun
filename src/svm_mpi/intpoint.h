/* -*-C++-*- */

#ifndef __INTPOINT_H_
#define __INTPOINT_H_

#include "matrix.h"
#include "optimizer.h"
//#include "kernel.h"
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


/*template <class T, class I>
void optimize_smw2(COptimizer &optimizer, CKernel<I,T> &kernel, CMatrix<T> &c, 
		   CMatrix<T> &in_pats,
		   CMatrix<T> &A, CMatrix<T> &b, CMatrix<T> &l, 
		   CMatrix<T> &u, const CMatrix<T> &r, const unsigned maxn,
		   CMatrix<T> &primal, CMatrix<T> &dual, const char **how);
*/

template <class T>
void optimize_smw2_core(COptimizer &optimizer, CMatrix<T> &c, CMatrix<T> &Z,
			CMatrix<T> &A, CMatrix<T> &b, CMatrix<T> &l, 
			CMatrix<T> &u, const CMatrix<T> &r,
			const unsigned treedepth,
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
			   IntpointResources *resources,
			   const unsigned treedepth, const char *xfile,
			   const char *dfile, CMatrix<T> &primal,
			   CMatrix<T> &dual, const char **how);
/*
  optimize_smw2mpi_core<double>(optimizer, *c, Z, *A, *b, *l, *u,
				*r, m_prime, m_last, my_rank,
				num_nodes, res, *primal,
				*dual, &how);				
*/

/* This one passes an implicit NULL to kernel in the above */
template <class I, class T>
void run_non_root_2mpi(const unsigned my_rank, const unsigned num_nodes);

void send_z_columns_double(MPI_Comm comm, double *data, const unsigned start_col
			   ,
                           const unsigned ncols, const unsigned nrows,
                           const int rank, const bool ishomogeneous) ;


#endif /* HAVE_MPI */

#endif /* ! __INTPOINT_H_ */
