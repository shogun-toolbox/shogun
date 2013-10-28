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
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK
#include <shogun/lib/common.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/base/Parallel.h>
#include <shogun/io/SGIO.h>

#include <pthread.h>

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
#define DSYGVX dsygvx
#define DSTEMR dstemr
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
#define DSYGVX dsygvx_
#define DSTEMR dstemr_
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
#ifdef HAVE_ACML
	DGETRI(N,A,lda,ipiv,&info);
#else
	double* work;
	int n=N;
	int LDA=lda;
	int lwork = -1;
	double work1 = 0;
	DGETRI(&n,A,&LDA,ipiv,&work1,&lwork,&info);
	lwork = (int) work1;
	work = SG_MALLOC(double, lwork);
	DGETRI(&n,A,&LDA,ipiv,work,&lwork,&info);
	SG_FREE(work);
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
int clapack_dpotrs(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
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
	ASSERT(*info==0)
	ASSERT(work1>0)
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
	double work1 = 0;
	int lworka = -1;
	DGESVD(&jobu, &jobvt, &m, &n, a, &lda, sing, u, &ldu, vt, &ldvt, &work1, &lworka, info);
	ASSERT(*info==0)
	lworka = (int) work1;
	double* worka = SG_MALLOC(double, lworka);
	DGESVD(&jobu, &jobvt, &m, &n, a, &lda, sing, u, &ldu, vt, &ldvt, worka, &lworka, info);
	SG_FREE(worka);
#endif
}
#undef DGESVD

void wrap_dgeqrf(int m, int n, double *a, int lda, double *tau, int *info)
{
#ifdef HAVE_ACML
	DGEQRF(m, n, a, lda, tau, info);
#else
	int lwork = -1;
	double work1 = 0;
	DGEQRF(&m, &n, a, &lda, tau, &work1, &lwork, info);
	ASSERT(*info==0)
	lwork = (int)work1;
	ASSERT(lwork>0)
	double* work = SG_MALLOC(double, lwork);
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
	double work1 = 0;
	DORGQR(&m, &n, &k, a, &lda, tau, &work1, &lwork, info);
	ASSERT(*info==0)
	lwork = (int)work1;
	ASSERT(lwork>0)
	double* work = SG_MALLOC(double, lwork);
	DORGQR(&m, &n, &k, a, &lda, tau, work, &lwork, info);
	SG_FREE(work);
#endif
}
#undef DORGQR

void wrap_dsyevr(char jobz, char uplo, int n, double *a, int lda, int il, int iu,
                 double *eigenvalues, double *eigenvectors, int *info)
{
	int m;
	double vl,vu;
	double abstol = 0.0;
	char I = 'I';
	int* isuppz = SG_MALLOC(int, n);
#ifdef HAVE_ACML
	DSYEVR(jobz,I,uplo,n,a,lda,vl,vu,il,iu,abstol,m,
	       eigenvalues,eigenvectors,n,isuppz,info);
#else
	int lwork = -1;
	int liwork = -1;
	double work1 = 0;
	int work2 = 0;
	DSYEVR(&jobz,&I,&uplo,&n,a,&lda,&vl,&vu,&il,&iu,&abstol,
               &m,eigenvalues,eigenvectors,&n,isuppz,
               &work1,&lwork,&work2,&liwork,info);
	ASSERT(*info==0)
	lwork = (int)work1;
	liwork = work2;
	double* work = SG_MALLOC(double, lwork);
	int* iwork = SG_MALLOC(int, liwork);
	DSYEVR(&jobz,&I,&uplo,&n,a,&lda,&vl,&vu,&il,&iu,&abstol,
               &m,eigenvalues,eigenvectors,&n,isuppz,
               work,&lwork,iwork,&liwork,info);
	ASSERT(*info==0)
	SG_FREE(work);
	SG_FREE(iwork);
	SG_FREE(isuppz);
#endif
}
#undef DSYEVR

void wrap_dsygvx(int itype, char jobz, char uplo, int n, double *a, int lda, double *b,
                 int ldb, int il, int iu, double *eigenvalues, double *eigenvectors, int *info)
{
	int m;
	double abstol = 0.0;
	double vl,vu;
	int* ifail = SG_MALLOC(int, n);
	char I = 'I';
#ifdef HAVE_ACML
	DSYGVX(itype,jobz,I,uplo,n,a,lda,b,ldb,vl,vu,
               il,iu,abstol,m,eigenvalues,
               eigenvectors,n,ifail,info);
#else
	int lwork = -1;
	double work1 = 0;
	int* iwork = SG_MALLOC(int, 5*n);
	DSYGVX(&itype,&jobz,&I,&uplo,&n,a,&lda,b,&ldb,&vl,&vu,
               &il,&iu,&abstol,&m,eigenvalues,eigenvectors,
               &n,&work1,&lwork,iwork,ifail,info);
	lwork = (int)work1;
	double* work = SG_MALLOC(double, lwork);
	DSYGVX(&itype,&jobz,&I,&uplo,&n,a,&lda,b,&ldb,&vl,&vu,
               &il,&iu,&abstol,&m,eigenvalues,eigenvectors,
               &n,work,&lwork,iwork,ifail,info);
	SG_FREE(work);
	SG_FREE(iwork);
	SG_FREE(ifail);
#endif
}
#undef DSYGVX

void wrap_dstemr(char jobz, char range, int n, double* diag, double *subdiag,
		double vl, double vu, int il, int iu, int* m, double* w, double* z__,
		int ldz, int nzc, int *isuppz, int tryrac, int *info)
{
#ifdef HAVE_ACML
	SG_SNOTIMPLEMENTED
#else
	int lwork=-1;
	int liwork=-1;
	double work1=0;
	int iwork1=0;
	DSTEMR(&jobz, &range, &n, diag, subdiag, &vl, &vu,
		&il, &iu, m, w, z__, &ldz, &nzc, isuppz, &tryrac,
		&work1, &lwork, &iwork1, &liwork, info);
	lwork=(int)work1;
	liwork=iwork1;
	double* work=SG_MALLOC(double, lwork);
	int* iwork=SG_MALLOC(int, liwork);
	DSTEMR(&jobz, &range, &n, diag, subdiag, &vl, &vu,
		&il, &iu, m, w, z__, &ldz, &nzc, isuppz, &tryrac,
		work, &lwork, iwork, &liwork, info);

	SG_FREE(work);
	SG_FREE(iwork);
#endif
}
#undef DSTEMR

}
#endif //HAVE_LAPACK
