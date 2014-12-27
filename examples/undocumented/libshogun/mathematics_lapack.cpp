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
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/linalg/eigsolver/EigenSolver.h>

using namespace shogun;

#ifdef HAVE_LAPACK

bool is_equal(float64_t a, float64_t b, float64_t eps)
{
	return CMath::abs(a-b)<=eps;
}

void test_ev()
{
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

	const CEigenSolver* ces;
	SGVector<float64_t> ev=ces->compute_eigenvectors(A);
	SGMatrix<float64_t>::display_matrix(A.matrix, A.num_rows, A.num_cols, "A");
	SGVector<float64_t>::display_vector(ev.vector, ev.vlen, "eigenvalues");

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
}

void test_matrix_multiply()
{
	index_t n=10;
	SGMatrix<float64_t> I=SGMatrix<float64_t>::create_identity_matrix(n,1.0);

	index_t m=4;
	SGMatrix<float64_t> A(n, m);
	SGVector<float64_t>::range_fill_vector(A.matrix, m*n);
	SGMatrix<float64_t>::display_matrix(I, "I");
	SGMatrix<float64_t>::transpose_matrix(A.matrix, A.num_rows, A.num_cols);
	SGMatrix<float64_t>::display_matrix(A, "A transposed");
	SGMatrix<float64_t>::transpose_matrix(A.matrix, A.num_rows, A.num_cols);
	SGMatrix<float64_t>::display_matrix(A, "A");

	SG_SPRINT("multiply A by I and check result\n");
	SGMatrix<float64_t> A2=SGMatrix<float64_t>::matrix_multiply(I, A);
	ASSERT(A2.num_rows==A.num_rows);
	ASSERT(A2.num_cols==A.num_cols);
	SGMatrix<float64_t>::display_matrix(A2);
	for (index_t i=0; i<A2.num_rows; ++i)
	{
		for (index_t j=0; j<A2.num_cols; ++j)
			ASSERT(A(i,j)==A2(i,j));
	}

	SG_SPRINT("multiply A by transposed I and check result\n");
	SGMatrix<float64_t> A3=SGMatrix<float64_t>::matrix_multiply(I, A, true);
	ASSERT(A3.num_rows==I.num_rows);
	ASSERT(A3.num_cols==A.num_cols);
	SGMatrix<float64_t>::display_matrix(A3);
	for (index_t i=0; i<A2.num_rows; ++i)
	{
		for (index_t j=0; j<A2.num_cols; ++j)
			ASSERT(A(i,j)==A3(i,j));
	}

	SG_SPRINT("multiply transposed A by I and check result\n");
	SGMatrix<float64_t> A4=SGMatrix<float64_t>::matrix_multiply(A, I, true, false);
	ASSERT(A4.num_rows==A.num_cols);
	ASSERT(A4.num_cols==I.num_cols);
	SGMatrix<float64_t>::display_matrix(A4);
	for (index_t i=0; i<A.num_rows; ++i)
	{
		for (index_t j=0; j<A.num_cols; ++j)
			ASSERT(A(i,j)==A4(j,i));
	}

	SG_SPRINT("multiply A by scaled I and check result\n");
	SGMatrix<float64_t> A5=SGMatrix<float64_t>::matrix_multiply(I, A, false, false, n);
	ASSERT(A5.num_rows==I.num_rows);
	ASSERT(A5.num_cols==A.num_cols);
	SGMatrix<float64_t>::display_matrix(A5);
	for (index_t i=0; i<A2.num_rows; ++i)
	{
		for (index_t j=0; j<A2.num_cols; ++j)
			ASSERT(n*A(i,j)==A5(i,j));
	}
}

void test_lapack()
{
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
		SG_SERROR("DSYGVX/SSYGVX failed with code %d\n",status);

	delete[] double_eigenvectors;

	// DGEQRF+DORGQR
	status = 0;
	double* double_tau = new double[N];
	wrap_dgeqrf(N,N,double_matrix,N,double_tau,&status);
	wrap_dorgqr(N,N,N,double_matrix,N,double_tau,&status);
	if (status!=0)
		SG_SERROR("DGEQRF/DORGQR failed with code %d\n",status);

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
		SG_SERROR("DGESVD failed with code %d\n",status);

	delete[] double_s;
	delete[] double_U;
	delete[] double_Vt;

	// DSYEV
	status = 0;
	wrap_dsyev('V','U',N,double_matrix,N,double_eigenvalues,&status);
	if (status!=0)
		SG_SERROR("DSYEV failed with code %d\n",status);

	delete[] double_eigenvalues;
	delete[] double_matrix;
}
#endif // HAVE_LAPACK

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

#ifdef HAVE_LAPACK
	SG_SPRINT("checking lapack\n");
	test_lapack();

	SG_SPRINT("compute_eigenvectors\n");
	test_ev();

	SG_SPRINT("matrix_multiply\n");
	test_matrix_multiply();
#endif // HAVE_LAPACK

	exit_shogun();
	return 0;
}


