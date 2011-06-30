/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "lib/arpack.h"
#include "lib/config.h"
#include <cblas.h>
#include "lib/common.h"
#include "lib/io.h"

using namespace shogun;

namespace shogun
{
void arpack_dsaupd(double* matrix, int n, int nev, char* which,
		   double* eigenvalues, double* eigenvectors, int& status)
{
	// init ARPACK's reverse communication parameter 
 	// (should be zero initially)
	int ido = 0;

	// specify that non-general eigenproblem will be solved
	// (Ax=lGx, where G=I)
	char bmat[2] = "I";

	// init tolerance (zero means machine precision)
	double tol = 0.0;

	// allocate array to hold residuals
	double* resid = new double[n];

	// set number of Lanczos basis vectors to be used
	// (with max(4*nev,n) sufficient for most tasks)
	int ncv = nev*4>n ? n : nev*4;

	// allocate array 'v' for dsaupd routine usage
	int ldv = n;
	double* v = new double[ldv*ncv];

	// init array for i/o params for routine
	int* iparam = new int[11];
	// specify shift strategy 
	iparam[0] = 1;
	// specify max number of iterations
	iparam[2] = 3*n;
	// set the computation mode 
	iparam[6] = 1;

	// init array indicating locations of vectors for routine callback
	int* ipntr = new int[11];

	// allocate workaround arrays
	double* workd = new double[3*n];
	int lworkl = ncv*(ncv+8);
	double* workl = new double[lworkl];

	// init info holding status (should be zero at first call)
	int info = 0;
	// main computation loop
	do 
	{
		// call ARPACK's dsaupd routine
		dsaupd_(&ido, bmat, &n, which, &nev, &tol, resid,
		        &ncv, v, &ldv, iparam, ipntr, workd, workl,
		        &lworkl, &info);

		// compute workd(ipntr(1))=matrix*workd(ipntr(0))
		if ((ido==1)||(ido==-1))
		{
			
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
			            n, 1, n,
				    1.0, matrix, n,
			            (workd+ipntr[0]-1), n,
			            0.0, (workd+ipntr[1]-1), n);
			
		}
	} while ((ido==1)||(ido==-1));
	// allocate select for dseupd
	int* select = new int[ncv];
	// allocate d to hold eigenvalues
	double* d = new double[2*ncv];

	// sigma for dseupd
	double sigma;
	// extract eigenpairs
	if (info<0)
	{
		// tell me tell me what you gonna do
		status = -1;
	}
	else 
	{
		// init ierr indicating dseupd possible errors
		int ierr;

		// specify that eigenvectors to be computed too		
		int rvec = 1;

		dseupd_(&rvec, "All", select, d, v, &ldv, &sigma, bmat,
		        &n, which, &nev, &tol, resid, &ncv, v, &ldv,
		        iparam, ipntr, workd, workl, &lworkl, &ierr);

		// TODO error check
	}
	for (int i=0; i<nev; i++)
	{
		eigenvalues[i] = d[i];

		for (int j=0; j<n; j++)
			eigenvectors[j*nev+i] = v[i*n+j];
	}
	// cleanup
	delete[] resid;
	delete[] v;
	delete[] iparam;
	delete[] ipntr;
	delete[] workd;
	delete[] workl;
	delete[] select;
	delete[] d;
};

}
