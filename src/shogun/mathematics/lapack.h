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
// double precision
int clapack_dpotrf(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
		const int N, double *A, const int lda);
int clapack_dposv(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
		const int N, const int NRHS, double *A, const int lda,
		double *B, const int ldb);
int clapack_dpotri(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
		const int N, double *A, const int LDA);
int clapack_dpotrs(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                   const int N, const int NRHS, double *A, const int lda,
                   double *B, const int ldb);
int clapack_dgetrf(const CBLAS_ORDER Order, const int M, const int N,
                   double *A, const int lda, int *ipiv);
int clapack_dgetri(const CBLAS_ORDER Order, const int N, double *A,
                   const int lda, int *ipiv);
int clapack_dgetrs(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE Transpose,
                   const int N, const int NRHS, double *A, const int lda,
                   int *ipiv, double *B, const int ldb);

// single precision
/*
int clapack_spotrf(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
		const int N, float *A, const int lda);
int clapack_sposv(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
		const int N, const int NRHS, float *A, const int lda,
		float *B, const int ldb);
int clapack_spotri(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
		const int N, float *A, const int LDA);
int clapack_spotrs(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                   const int N, const int NRHS, float *A, const int lda,
                   float *B, const int ldb);
int clapack_sgetrf(const CBLAS_ORDER Order, const int M, const int N,
                   float *A, const int lda, int *ipiv);
int clapack_sgetri(const CBLAS_ORDER Order, const int N, float *A,
                   const int lda, int *ipiv);
int clapack_sgetrs(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE Transpose,
                   const int N, const int NRHS, float *A, const int lda,
                   int *ipiv, float *B, const int ldb);
*/
#endif

namespace shogun
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
// double precision
void wrap_dsyev(char jobz, char uplo, int n, double *a, int lda, 
		double *w, int *info);
void wrap_dgesvd(char jobu, char jobvt, int m, int n, double *a, int lda, 
		double *sing, double *u, int ldu, double *vt, int ldvt, 
		int *info);
void wrap_dgeqrf(int m, int n, double *a, int lda, double *tau, int *info);
void wrap_dorgqr(int m, int n, int k, double *a, int lda, double *tau, int *info);
void wrap_dsyevr(char jobz, char uplo, int n, double *a, int lda, int il, int iu, 
                 double *eigenvalues, double *eigenvectors, int *info);
void wrap_dsygvx(int itype, char jobz, char uplo, int n, double *a, int lda, double *b,
                 int ldb, int il, int iu, double *eigenvalues, double *eigenvectors, int *info);

// single precision
void wrap_ssyev(char jobz, char uplo, int n, float *a, int lda, 
		float *w, int *info);
void wrap_sgesvd(char jobu, char jobvt, int m, int n, float *a, int lda, 
		float *sing, float *u, int ldu, float *vt, int ldvt, 
		int *info);
void wrap_sgeqrf(int m, int n, float *a, int lda, float *tau, int *info);
void wrap_sorgqr(int m, int n, int k, float *a, int lda, float *tau, int *info);
void wrap_ssyevr(char jobz, char uplo, int n, float *a, int lda, int il, int iu, 
                 float *eigenvalues, float *eigenvectors, int *info);
void wrap_ssygvx(int itype, char jobz, char uplo, int n, float *a, int lda, float *b,
                 int ldb, int il, int iu, float *eigenvalues, float *eigenvectors, int *info);
#endif
}

// only MKL, ACML and Mac OS vector library provide a header file for the lapack routines
#if !defined(HAVE_ACML) && !defined(HAVE_MKL) && !defined(HAVE_MVEC)
// double precision
int dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);
int dgesvd_(char* jobu, char* jobvt, int* m, int* n, double* a, int* lda,
		double* s, double* u, int* ldu, double* vt, int* ldvt, double* work,
		int* lwork, int* info);
int dposv_(const char *uplo, const int *n, const int *nrhs, double *a, const int *lda, double *b, const int *ldb, int *info);
int dpotrf_(const char *uplo, int *n, double *a, int * lda, int *info);
int dpotri_(const char *uplo, int *n, double *a, int * lda, int *info);
int dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
int dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
int dgeqrf_(int*, int*, double*, int*, double*, double*, int*, int*);
int dorgqr_(int*, int*, int*, double*, int*, double*, double*, int*, int*);
int dsyevr_(const char*, const char*, const char*, int*, double*, int*, 
            double*, double*, int*, int*, double*, int*, double*, double*,
            int*, int*, double*, int*, int*, int*, int*);
int dgetrs_(const char*, int*, int*, double*, int*, int*, double*, int*, int*);
int dpotrs_(const char*, int*, int*, double*, int*, double*, int*, int*);
int dsygvx_(int*, const char*, const char*, const char*, int*, double*, int*,
            double*, int*, double* , double*, int*, int*, double*,
            int*, double*, double*, int*, double*, int*, int*, int*, int*);

// single precision
int ssyev_(char*, char*, int*, float*, int*, float*, float*, int*, int*);
int sgesvd_(char* jobu, char* jobvt, int* m, int* n, float* a, int* lda,
		float* s, float* u, int* ldu, float* vt, int* ldvt, float* work,
		int* lwork, int* info);
int sposv_(const char *uplo, const int *n, const int *nrhs, float *a, const int *lda, float *b, const int *ldb, int *info);
int spotrf_(const char *uplo, int *n, float *a, int * lda, int *info);
int spotri_(const char *uplo, int *n, float *a, int * lda, int *info);
int sgetri_(int *n, float *a, int *lda, int *ipiv, float *work, int *lwork, int *info);
int sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info);
int sgeqrf_(int*, int*, float*, int*, float*, float*, int*, int*);
int sorgqr_(int*, int*, int*, float*, int*, float*, float*, int*, int*);
int ssyevr_(const char*, const char*, const char*, int*, float*, int*, 
            float*, float*, int*, int*, float*, int*, float*, float*,
            int*, int*, float*, int*, int*, int*, int*);
int sgetrs_(const char*, int*, int*, float*, int*, int*, float*, int*, int*);
int spotrs_(const char*, int*, int*, float*, int*, float*, int*, int*);
int ssygvx_(int*, const char*, const char*, const char*, int*, float*, int*,
            float*, int*, float* , float*, int*, int*, float*,
            int*, float*, float*, int*, float*, int*, int*, int*, int*);

#endif
}

#endif //HAVE_LAPACK
#endif //_LAPACK_H__
