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
#include <shogun/mathematics/arpack.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun();

	int N = 100;
	int nev = 2;

	double* matrix = new double[N*N];

	double* eigenvalues = new double[nev];
	double* eigenvectors = new double[nev*N];

	for (int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
			matrix[i*N+j] = i*i+j*j;
	}

	int status = 0;
	arpack_dsaeupd_wrap(matrix, NULL, N, 2, "LM", 1, false, 0.0, 0.0,
	                    eigenvalues, eigenvectors, status);
	if (status!=0)
		return -1;

	arpack_dsaeupd_wrap(matrix, NULL, N, 2, "BE", 3, false, 1.0, 0.0,
	                    eigenvalues, eigenvectors, status);
	if (status!=0)
		return -1;


	delete[] eigenvalues;
	delete[] eigenvectors;
	delete[] matrix;

	exit_shogun();
	return 0;
}
