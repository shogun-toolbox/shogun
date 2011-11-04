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
#include <shogun/mathematics/arpack.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun();

#ifdef HAVE_ARPACK
	int N = 100;
	int nev = 2;

	double* double_matrix = new double[N*N];
	double* rhs_double_diag = new double[N];
	double* double_eigenvalues = new double[nev];
	double* double_eigenvectors = new double[nev*N];

	for (int i=0; i<N; i++)
	{
		rhs_double_diag[i] = 1.0;
		for (int j=0; j<N; j++)
		{
			double_matrix[i*N+j] = i*i+j*j;
		}
	}

	int status = 0;
	arpack_dsxupd(double_matrix, NULL, false, N, 2, "LM", false, 1, false, 0.0, 0.0,
	              double_eigenvalues, double_eigenvectors, status);
	if (status!=0)
		return -1;

	arpack_dsxupd(double_matrix, NULL, false, N, 2, "BE", false, 3, false, 1.0, 0.0,
	              double_eigenvalues, double_eigenvectors, status);
	if (status!=0)
		return -1;

	arpack_dsxupd(double_matrix, rhs_double_diag, true, N, 2, "SM", false, 3, false, 0.0, 0.0,
	              double_eigenvalues, double_eigenvectors, status);
	if (status!=0)
		return -1;

	delete[] double_eigenvalues;
	delete[] double_eigenvectors;
	delete[] double_matrix;
	delete[] rhs_double_diag;
#endif // HAVE_ARPACK

	exit_shogun();
	return 0;
}
