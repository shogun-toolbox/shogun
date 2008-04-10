/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2006-2007 Mikio L. Braun
 * Written (W) 2008 Jochen Garcke
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LAPACK_H__
#define _LAPACK_H__

#include "lib/config.h"
#include "lib/common.h"

#ifdef HAVE_LAPACK
extern "C" {
#ifdef HAVE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#ifdef HAVE_ATLAS
#include <clapack.h>
#endif
}

void wrap_dsyev(char jobz, char uplo, int n, double *a, int lda, 
		double *w, int *info);
void wrap_dgesvd(char jobu, char jobvt, int m, int n, double *a, int lda, 
		double *sing, double *u, int ldu, double *vt, int ldvt, 
		int *info);

#ifdef HAVE_ATLAS
// ATLAS does not provide a header file for the lapack routines
extern "C" {
	int dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);
	int dgesvd_(char* jobu, char* jobvt, int* m, int* n, double* a, int* lda,
			double* s, double* u, int* ldu, double* vt, int* ldvt, double* work,
			int* lwork, int* info);
}
#else //NON_ATLAS libs
#ifdef HAVE_ACML
#include <acml.h>
#endif
#ifdef HAVE_MKL
#include <mkl_lapack.h>
#endif
#ifdef DARWIN

#endif 

int clapack_dpotrf(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
		const int N, double *A, const int lda);
int clapack_dposv(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
		const int N, const int NRHS, double *A, const int lda,
		double *B, const int ldb);
#endif // NON_ATLAS libs
#endif //HAVE_LAPACK
#endif //_LAPACK_H__
