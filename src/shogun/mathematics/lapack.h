/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2006-2007 Mikio L. Braun
 * Written (W) 2008 Jochen Garcke
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LAPACK_H__
#define _LAPACK_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>

#ifdef HAVE_LAPACK

extern "C" {

#ifdef HAVE_MKL
#include <mkl_cblas.h>
#include <mkl_lapack.h>
#elif defined(HAVE_MVEC)
#include </System/Library/Frameworks/vecLib.framework/Headers/cblas.h>
#include </System/Library/Frameworks/vecLib.framework/Headers/clapack.h>
#else
#include <cblas.h>
#endif

#ifdef HAVE_ACML
#include <acml.h>
#endif

#ifdef HAVE_ATLAS
#include <clapack.h>
#else
// ACML and MKL do not provide clapack_* routines
int clapack_dpotrf(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
		const int N, double *A, const int lda);
int clapack_dposv(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
		const int N, const int NRHS, double *A, const int lda,
		double *B, const int ldb);
int clapack_dpotri(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
		const int N, double *A, const int LDA);
int clapack_dgetrf(const CBLAS_ORDER Order, const int M, const int N,
                   double *A, const int lda, int *ipiv);
int clapack_dgetri(const CBLAS_ORDER Order, const int N, double *A,
                   const int lda, int *ipiv);
#endif

namespace shogun
{
void wrap_dsyev(char jobz, char uplo, int n, double *a, int lda, 
		double *w, int *info);
void wrap_dgesvd(char jobu, char jobvt, int m, int n, double *a, int lda, 
		double *sing, double *u, int ldu, double *vt, int ldvt, 
		int *info);
}

// only MKL, ACML and Mac OS vector library provide a header file for the lapack routines
#if !defined(HAVE_ACML) && !defined(HAVE_MKL) && !defined(HAVE_MVEC)
int dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);
int dgesvd_(char* jobu, char* jobvt, int* m, int* n, double* a, int* lda,
		double* s, double* u, int* ldu, double* vt, int* ldvt, double* work,
		int* lwork, int* info);
int dposv_(const char *uplo, const int *n, const int *nrhs, double *a, const int *lda, double *b, const int *ldb, int *info);
int dpotrf_(const char *uplo, int *n, double *a, int * lda, int *info);
int dpotri_(const char *uplo, int *n, double *a, int * lda, int *info);
int dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
int dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
#endif
}

#endif //HAVE_LAPACK
#endif //_LAPACK_H__
