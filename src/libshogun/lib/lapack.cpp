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

#include "lib/config.h"

#ifdef HAVE_LAPACK
#include "lib/lapack.h"
#include "lib/common.h"
#include "lib/io.h"

using namespace shogun;

#if defined(HAVE_MKL) || defined(HAVE_ACML) 
#define DSYEV dsyev
#define DGESVD dgesvd
#define DPOSV dposv
#define DPOTRF dpotrf
#else
#define DSYEV dsyev_
#define DGESVD dgesvd_
#define DPOSV dposv_
#define DPOTRF dpotrf_
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
	else
		if (Uplo==CblasLower)
			uplo='L';
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
	else
		if (Uplo==CblasLower)
			uplo='L';
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
#endif //HAVE_ATLAS

/*
 * Wrapper files for LAPACK if there isn't a clapack interface
 * 
 */

namespace shogun
{
/*  DSYEV computes all eigenvalues and, optionally, eigenvectors of a
 *  real symmetric matrix A.
 */
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
	double* work=new double[lwork];
	DSYEV(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
	delete[] work;
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
	double* work=new double[lwork];
	DGESVD(&jobu, &jobvt, &m, &n, a, &lda, sing, u, &ldu, vt, &ldvt, work, &lwork, info);
	delete[] work;
#endif
}
}
#undef DGESVD
#endif //HAVE_LAPACK
