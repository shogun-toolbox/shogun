#include "lib/common.h"
#ifdef HAVE_LAPACK
#include "lib/lapack.h"
#include <assert.h>

#if defined(HAVE_LAPACK) && defined(DARWIN)
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
#endif
