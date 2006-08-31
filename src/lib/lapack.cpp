/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#if defined(HAVE_LAPACK) && defined(DARWIN)
#include "lib/lapack.h"
#include <assert.h>

int clapack_dpotrf(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
			                   const int N, double *A, const int LDA)
{
	assert(Order==CblasColMajor);
        //call dgemm ( 'T', 'T', alpha, B, ldb, A, lda, beta, C, ldc )
	char uplo = 'U';
	int info=0;
	int n=N;
	int lda=LDA;

	if (Uplo==CblasLower)
		uplo='L';
	dpotrf_(&uplo, &n, A, &lda, &info);
	return info;
}
#endif
