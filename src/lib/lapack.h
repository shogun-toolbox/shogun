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

#ifndef _LAPACK_H__
#define _LAPACK_H__

#include "lib/common.h"

#ifdef HAVE_LAPACK

#ifdef DARWIN
#include <cblas.h>
int clapack_dpotrf(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
			                   const int N, double *A, const int lda);
#endif //DARWIN


extern "C" {
int dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);
int dpotrf_ (char *uplo, int *n, double *a, int *lda, int *info);
}
#endif //HAVE_LAPACK
#endif //_LAPACK_H__
