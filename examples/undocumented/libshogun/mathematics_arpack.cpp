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
	float* float_matrix = new float[N*N];

	double* rhs_double_diag = new double[N];
	float* rhs_float_diag = new float[N];

	double* double_eigenvalues = new double[nev];
	float* float_eigenvalues = new float[nev];

	double* double_eigenvectors = new double[nev*N];
	float* float_eigenvectors = new float[nev*N];

	for (int i=0; i<N; i++)
	{
		rhs_double_diag[i] = 1.0;
		rhs_float_diag[i] = 1.0;
		for (int j=0; j<N; j++)
		{
			double_matrix[i*N+j] = i*i+j*j;
			float_matrix[i*N+j] = i*i+j*j;
		}
	}

	int status = 0;
	arpack_xsxupd<double>(double_matrix, NULL, N, 2, "LM", 1, false, 0.0, 0.0,
	                      double_eigenvalues, double_eigenvectors, status);
	arpack_xsxupd<float>(float_matrix, NULL, N, 2, "LM", 1, false, 0.0, 0.0,
	                     float_eigenvalues, float_eigenvectors, status);
	if (status!=0)
		return -1;

	arpack_xsxupd<double>(double_matrix, NULL, N, 2, "BE", 3, false, 1.0, 0.0,
	                      double_eigenvalues, double_eigenvectors, status);
	arpack_xsxupd<float>(float_matrix, NULL, N, 2, "BE", 3, false, 1.0, 0.0,
	                     float_eigenvalues, float_eigenvectors, status);
	if (status!=0)
		return -1;

	arpack_xsxupd<double>(double_matrix, rhs_double_diag, N, 2, "SM", 3, false, 0.0, 0.0,
	                      double_eigenvalues, double_eigenvectors, status);
	arpack_xsxupd<float>(float_matrix, rhs_float_diag, N, 2, "SM", 3, false, 0.0, 0.0,
	                     float_eigenvalues, float_eigenvectors, status);
	if (status!=0)
		return -1;

	delete[] double_eigenvalues;
	delete[] double_eigenvectors;
	delete[] double_matrix;
	delete[] rhs_double_diag;
	delete[] float_eigenvalues;
	delete[] float_eigenvectors;
	delete[] float_matrix;
	delete[] rhs_float_diag;
#endif // HAVE_ARPACK

	exit_shogun();
	return 0;
}
