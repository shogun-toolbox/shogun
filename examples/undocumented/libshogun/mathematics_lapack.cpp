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
	double* matrix = new double[N*N];
	// for storing eigenpairs
	double* eigenvalues = new double[N];
	double* eigenvectors = new double[N*N];
	// for SVD
	double* U = new double[N*N];
	double* s = new double[N*N];
	double* Vt = new double[N*N];
	// status (should be zero)
	int status;

	// DSYGVX
	for (int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
			matrix[i*N+j] = i*i+j*j;
	}	
	status = 0;
	wrap_dsygvx(1,'V','U',N,matrix,N,matrix,N,1,3,eigenvalues,eigenvectors,&status);
	if (status!=0)
		return -1;
	delete[] eigenvectors;


	// DGEQRF+DORGQR
	status = 0;
	double* tau = new double[N];
	wrap_dgeqrf(N,N,matrix,N,tau,&status);
	wrap_dorgqr(N,N,N,matrix,N,tau,&status);
	if (status!=0)
		return -1;
	delete[] tau;


	// DGESVD
	for (int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
			matrix[i*N+j] = i*i+j*j;
	}
	status = 0;
	wrap_dgesvd('A','A',N,N,matrix,N,s,U,N,Vt,N,&status);
	if (status!=0)
		return -1;
	delete[] s;
	delete[] U;
	delete[] Vt;


	// DSYEV
	for (int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
			matrix[i*N+j] = i*i+j*j;
	}
	status = 0;
	wrap_dsyev('V','U',N,matrix,N,eigenvalues,&status);
	if (status!=0)
		return -1;
	delete[] eigenvalues;
	delete[] matrix;

	#endif // HAVE_LAPACK

	exit_shogun();
	return 0;
}
