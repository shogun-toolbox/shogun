/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/lapack.h>

using namespace shogun;

bool is_equal(float64_t a, float64_t b, float64_t eps)
{
	return CMath::abs(a-b)<=eps;
}

void test_ev()
{
#ifdef HAVE_LAPACK

	SGMatrix<float64_t> A(3,3);
	A(0,0)=0;
	A(0,1)=1;
	A(0,2)=0;
	A(1,0)=1;
	A(1,1)=0;
	A(1,2)=1;
	A(1,0)=0;
	A(2,1)=1;
	A(2,2)=0;

	SGVector<float64_t> ev=CMath::compute_eigenvectors(A);
	CMath::display_matrix(A.matrix, A.num_rows, A.num_cols, "A");
	CMath::display_vector(ev.vector, ev.vlen, "eigenvalues");

	float64_t sqrt22=CMath::sqrt(2.0)/2.0;
	float64_t eps=10E-16;

	/* check for correct eigenvectors */
	ASSERT(is_equal(A(0,0), 0.5, eps));
	ASSERT(is_equal(A(0,1), -sqrt22, eps));
	ASSERT(is_equal(A(0,2), 0.5, eps));

	ASSERT(is_equal(A(1,0), -sqrt22, eps));
	ASSERT(is_equal(A(1,1), 0, eps));
	ASSERT(is_equal(A(1,2), sqrt22, eps));

	ASSERT(is_equal(A(2,0), 0.5, eps));
	ASSERT(is_equal(A(2,1), sqrt22, eps));
	ASSERT(is_equal(A(2,2), 0.5, eps));

	/* check for correct eigenvalues */
	ASSERT(is_equal(ev[0], -sqrt22*2, eps));
	ASSERT(is_equal(ev[1], 0, eps));
	ASSERT(is_equal(ev[2], sqrt22*2, eps));

#endif /* HAVE_LAPACK */
}

void test_lapack()
{
#ifdef HAVE_LAPACK
	// size of square matrix
	int N = 100;

	// square matrix
	double* double_matrix = new double[N*N];
	// for storing eigenpairs
	double* double_eigenvalues = new double[N];
	double* double_eigenvectors = new double[N*N];
	// for SVD
	double* double_U = new double[N*N];
	double* double_s = new double[N];
	double* double_Vt = new double[N*N];
	// status (should be zero)
	int status;

	// DSYGVX
	for (int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
			double_matrix[i*N+j] = ((double)(i-j))/(i+j+1);

		double_matrix[i*N+i] += 100;
	}
	status = 0;
	wrap_dsygvx(1,'V','U',N,double_matrix,N,double_matrix,N,1,3,double_eigenvalues,double_eigenvectors,&status);
	if (status!=0)
	{
		printf("DSYGVX/SSYGVX failed with code %d\n",status);
		ASSERT(false);
	}
	delete[] double_eigenvectors;

	// DGEQRF+DORGQR
	status = 0;
	double* double_tau = new double[N];
	wrap_dgeqrf(N,N,double_matrix,N,double_tau,&status);
	wrap_dorgqr(N,N,N,double_matrix,N,double_tau,&status);
	if (status!=0)
	{
		printf("DGEQRF/DORGQR failed with code %d\n",status);
		ASSERT(false);
	}
	delete[] double_tau;

	// DGESVD
	for (int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
			double_matrix[i*N+j] = i*i+j*j;
	}
	status = 0;
	wrap_dgesvd('A','A',N,N,double_matrix,N,double_s,double_U,N,double_Vt,N,&status);
	if (status!=0)
	{
		printf("DGESVD failed with code %d\n",status);
		ASSERT(false);
	}
	delete[] double_s;
	delete[] double_U;
	delete[] double_Vt;

	// DSYEV
	status = 0;
	wrap_dsyev('V','U',N,double_matrix,N,double_eigenvalues,&status);
	if (status!=0)
	{
		printf("DSYEV failed with code %d\n",status);
		ASSERT(false);
	}
	delete[] double_eigenvalues;
	delete[] double_matrix;

	#endif // HAVE_LAPACK
}

int main(int argc, char** argv)
{
	init_shogun();

	test_lapack();
	test_ev();

	exit_shogun();
	return 0;
}

