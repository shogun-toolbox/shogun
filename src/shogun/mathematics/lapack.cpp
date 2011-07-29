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

#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

#if defined(HAVE_MKL) || defined(HAVE_ACML) 
#define DSYEV dsyev
#define DGESVD dgesvd
#define DPOSV dposv
#define DPOTRF dpotrf
#define DPOTRI dpotri
#define DGETRI dgetri
#define DGETRF dgetrf
#define DGEQRF dgeqrf
#define DORGQR dorgqr
#define DSYEVR dsyevr
#define DPOTRS dpotrs
#define DGETRS dgetrs
#else
#define DSYEV dsyev_
#define DGESVD dgesvd_
#define DPOSV dposv_
#define DPOTRF dpotrf_
#define DPOTRI dpotri_
#define DGETRI dgetri_
#define DGETRF dgetrf_
#define DGEQRF dgeqrf_
#define DORGQR dorgqr_
#define DSYEVR dsyevr_
#define DGETRS dgetrs_
#define DPOTRS dpotrs_
#endif

#ifndef HAVE_ATLAS
int clapack_dpotrf(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
		const int N, double *A, const int LDA)
{
	char uplo = 'U';
	int info = 0;
	if (Order==CblasRowMajor)
	{//A is symmetric, we switch Uplo to get result for CblasRowMajor
		if (Uplo==CblasUpper)
			uplo='L';
	}
	else if (Uplo==CblasLower)
	{
		uplo='L';
	}
#ifdef HAVE_ACML
	DPOTRF(uplo, N, A, LDA, &info);
#else
	int n=N;
	int lda=LDA;
	DPOTRF(&uplo, &n, A, &lda, &info);
#endif
	return info;
}
#undef DPOTRF

int clapack_dpotri(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
		const int N, double *A, const int LDA)
{
	char uplo = 'U';
	int info = 0;
	if (Order==CblasRowMajor)
	{//A is symmetric, we switch Uplo to get result for CblasRowMajor
		if (Uplo==CblasUpper)
			uplo='L';
	}
	else if (Uplo==CblasLower)
	{
		uplo='L';
	}
#ifdef HAVE_ACML
	DPOTRI(uplo, N, A, LDA, &info);
#else
	int n=N;
	int lda=LDA;
	DPOTRI(&uplo, &n, A, &lda, &info);
#endif
	return info;
}
#undef DPOTRI

/* DPOSV computes the solution to a real system of linear equations
 * A * X = B,
 * where A is an N-by-N symmetric positive definite matrix and X and B
 * are N-by-NRHS matrices
 */
int clapack_dposv(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
		const int N, const int NRHS, double *A, const int lda,
		double *B, const int ldb)
{
	char uplo = 'U';
	int info=0;
	if (Order==CblasRowMajor)
	{//A is symmetric, we switch Uplo to achieve CblasColMajor
		if (Uplo==CblasUpper)
			uplo='L';
	}
	else if (Uplo==CblasLower)
	{
		uplo='L';
	}
#ifdef HAVE_ACML
	DPOSV(uplo,N,NRHS,A,lda,B,ldb,&info);
#else
	int n=N;
	int nrhs=NRHS;
	int LDA=lda;
	int LDB=ldb;
	DPOSV(&uplo, &n, &nrhs, A, &LDA, B, &LDB, &info);
#endif
	return info;
}
#undef DPOSV

int clapack_dgetrf(const CBLAS_ORDER Order, const int M, const int N,
                   double *A, const int lda, int *ipiv)
{
	// no rowmajor?
	int info=0;
#ifdef HAVE_ACML
	DGETRF(M,N,A,lda,ipiv,&info);
#else
	int m=M;
	int n=N;
	int LDA=lda;
	DGETRF(&m,&n,A,&LDA,ipiv,&info);
#endif
	return info;
}
#undef DGETRF

// order not supported (yet?)
int clapack_dgetri(const CBLAS_ORDER Order, const int N, double *A,
                   const int lda, int* ipiv)
{
	int info=0;
	double* work = SG_MALLOC(double, 1);
#ifdef HAVE_ACML
	int lwork = -1;
	DGETRI(N,A,lda,ipiv,work,lwork,&info);
	lwork = (int) work[0];
	SG_FREE(work);
	work = SG_MALLOC(double, lwork);
	DGETRI(N,A,lda,ipiv,work,lwork,&info);
#else
	int n=N;
	int LDA=lda;
	int lwork = -1;
	DGETRI(&n,A,&LDA,ipiv,work,&lwork,&info);
	lwork = (int) work[0];
	SG_FREE(work);
	work = SG_MALLOC(double, lwork);
	DGETRI(&n,A,&LDA,ipiv,work,&lwork,&info);
#endif
	return info;
}
#undef DGETRI

// order not supported (yet?)
int clapack_dgetrs(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE Transpose,
                   const int N, const int NRHS, double *A, const int lda, 
                   int *ipiv, double *B, const int ldb)
{
	int info = 0;
	char trans = 'N';
	if (Transpose==CblasTrans) 
	{
		trans = 'T';
	}
#ifdef HAVE_ACML
	DGETRS(trans,N,NRHS,A,lda,ipiv,B,ldb,info);
#else
	int n=N;
	int nrhs=NRHS;
	int LDA=lda;
	int LDB=ldb;
	DGETRS(&trans,&n,&nrhs,A,&LDA,ipiv,B,&LDB,&info);
#endif
	return info;
}
#undef DGETRS

// order not supported (yet?)
int clapack_dpotrs(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                   const int N, const int NRHS, double *A, const int lda,
                   double *B, const int ldb)
{
	int info=0;
	char uplo = 'U';
	if (Uplo==CblasLower)
	{
		uplo = 'L';
	}
#ifdef HAVE_ACML
	DPOTRS(uplo,N,NRHS,A,lda,B,ldb,info);
#else
	int n=N;
	int nrhs=NRHS;
	int LDA=lda;
	int LDB=ldb;
	DPOTRS(&uplo,&n,&nrhs,A,&LDA,B,&LDB,&info);
#endif
	return info;
}
#undef DPOTRS

#endif //HAVE_ATLAS

namespace shogun
{

void wrap_dsyev(char jobz, char uplo, int n, double *a, int lda, double *w, int *info)
{
#ifdef HAVE_ACML
	DSYEV(jobz, uplo, n, a, lda, w, info);
#else
	int lwork=-1;
	double work1;
	DSYEV(&jobz, &uplo, &n, a, &lda, w, &work1, &lwork, info);
	ASSERT(*info==0);
	ASSERT(work1>0);
	lwork=(int) work1;
	double* work=SG_MALLOC(double, lwork);
	DSYEV(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
	SG_FREE(work);
#endif
}
#undef DSYEV

void wrap_dgesvd(char jobu, char jobvt, int m, int n, double *a, int lda, double *sing, 
		double *u, int ldu, double *vt, int ldvt, int *info)
{
#ifdef HAVE_ACML
	DGESVD(jobu, jobvt, m, n, a, lda, sing, u, ldu, vt, ldvt, info);
#else
	int lwork=-1;
	double work1;
	DGESVD(&jobu, &jobvt, &m, &n, a, &lda, sing, u, &ldu, vt, &ldvt, &work1, &lwork, info);
	ASSERT(*info==0);
	ASSERT(work1>0);
	lwork=(int) work1;
	double* work=SG_MALLOC(double, lwork);
	DGESVD(&jobu, &jobvt, &m, &n, a, &lda, sing, u, &ldu, vt, &ldvt, work, &lwork, info);
	SG_FREE(work);
#endif
}
#undef DGESVD

void wrap_dgeqrf(int m, int n, double *a, int lda, double *tau, int *info)
{
#ifdef HAVE_ACML
	DGEQRF(m, n, a, lda, tau, info);
#else
	int lwork = -1;
	double* work = SG_MALLOC(double, 1);
	DGEQRF(&m, &n, a, &lda, tau, work, &lwork, info);
	ASSERT(*info==0);
	lwork = (int) work[0];
	ASSERT(lwork>0)
	SG_FREE(work);
	work = SG_MALLOC(double, lwork);
	DGEQRF(&m, &n, a, &lda, tau, work, &lwork, info);
	SG_FREE(work);
#endif
}
#undef DGEQRF

void wrap_dorgqr(int m, int n, int k, double *a, int lda, double *tau, int *info)
{
#ifdef HAVE_ACML
	DORGQR(m, n, k, a, lda, tau, info);
#else
	int lwork = -1;
	double* work = SG_MALLOC(double, 1);
	DORGQR(&m, &n, &k, a, &lda, tau, work, &lwork, info);
	ASSERT(*info==0);
	lwork = (int) work[0];
	ASSERT(lwork>0);
	SG_FREE(work);
	work = SG_MALLOC(double, lwork);
	DORGQR(&m, &n, &k, a, &lda, tau, work, &lwork, info);
	SG_FREE(work);
#endif
}
#undef DORGQR

void wrap_dsyevr(char jobz, char uplo, int n, double *a, int lda, int il, int ul, 
                 double *eigenvalues, double *eigenvectors, int *info)
{
	int m;
	double vl,vu; 
	double abstol = 0.0;
	char I = 'I';
	int* isuppz = SG_MALLOC(int, n);
#ifdef HAVE_ACML
	DSYEVR(jobz,I,uplo,n,a,lda,vl,vu,il,ul,abstol,m,
	       eigenvalues,eigenvectors,n,isuppz,info);
#else
	int lwork = -1;
	int liwork = -1;
	double* work = SG_MALLOC(double, 1);
	int* iwork = SG_MALLOC(int, 1);
	DSYEVR(&jobz,&I,&uplo,&n,a,&lda,&vl,&vu,&il,&ul,&abstol,
               &m,eigenvalues,eigenvectors,&n,isuppz,
               work,&lwork,iwork,&liwork,info);
	ASSERT(*info==0);
	lwork = (int)work[0];
	liwork = iwork[0];
	SG_FREE(work);
	SG_FREE(iwork);
	work = SG_MALLOC(double, lwork);
	iwork = SG_MALLOC(int, liwork);
	DSYEVR(&jobz,&I,&uplo,&n,a,&lda,&vl,&vu,&il,&ul,&abstol,
               &m,eigenvalues,eigenvectors,&n,isuppz,
               work,&lwork,iwork,&liwork,info);
	ASSERT(*info==0);
	SG_FREE(work);
	SG_FREE(iwork);
	SG_FREE(isuppz);
#endif
}
#undef DSYEVR

}
#endif //HAVE_LAPACK
