/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/lapack.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun();

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
		return -1;
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
		return -1;
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
		return -1;
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
		return -1;
	}
	delete[] double_eigenvalues;
	delete[] double_matrix;

	#endif // HAVE_LAPACK

	exit_shogun();
	return 0;
}
