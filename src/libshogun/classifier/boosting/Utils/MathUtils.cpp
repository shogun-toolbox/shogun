/*
*
*    MultiBoost - Multi-purpose boosting package
*
*    Copyright (C) 2010   AppStat group
*                         Laboratoire de l'Accelerateur Lineaire
*                         Universite Paris-Sud, 11, CNRS
*
*    This file is part of the MultiBoost library
*
*    This library is free software; you can redistribute it
*    and/or modify it under the terms of the GNU General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    General Public License for more details.
*
*    You should have received a copy of the GNU General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 51 Franklin St, 5th Floor, Boston, MA 02110-1301 USA
*
*    Contact: Balazs Kegl (balazs.kegl@gmail.com)
*             Norman Casagrande (nova77@gmail.com)
*             Robert Busa-Fekete (busarobi@gmail.com)
*
*    For more information and up-to-date version, please visit
*
*                       http://www.multiboost.org/
*
*/

//#include "blaswrap.h"
//#include "f2c.h"

#define DIM 1024
#define WSIZE DIM * DIM

typedef int integer;
typedef double doublereal;
typedef float real;

////for MDL
static doublereal       work[WSIZE];
static integer          ipiv[WSIZE];

//////////////////////////////////////////////////////////////////////////////////////////////
// extern for lapack routines
extern "C" /* Subroutine */ int dgetri_(integer *n, doublereal *a, integer *lda, integer 
	*ipiv, doublereal *work, integer *lwork, integer *info);
extern "C" /* Subroutine */ int sgemm_(char *transa, char *transb, integer *m, integer *
	n, integer *k, real *alpha, real *a, integer *lda, real *b, integer *
	ldb, real *beta, real *c__, integer *ldc);
extern "C" /* Subroutine */ int dgetrf_(integer *m, integer *n, doublereal *a, integer *
	lda, integer *ipiv, integer *info);
extern "C" void solveEquationSystem( double* X, double* b, int* N );

int matrixInverse( integer *n, doublereal *a );
void solveEquationSystem( double* X, double* b, int* N );




void solveEquationSystem( double* X, double* b, int* N ) {
	matrixInverse( (integer*) N, (doublereal*) X );
	double* sum = new double[ *N ];
	for( int i = 0; i < *N; i++ ) {
		sum[i] = 0.0;
		for( int j = 0; j < *N; j++ ) {
			sum[i] += (X[(i*(*N))+j] * b[j]);
		}
	}
	for( int i = 0; i < *N; i++ ) {
		b[i] = sum[i];
	}
	
	delete [] sum;
}


int matrixInverse( integer *n, doublereal *a ) {
    integer info;
    static integer lwork = WSIZE;
    
	dgetrf_( n, n, a, n, ipiv, &info);
    dgetri_( n, a, n, ipiv, work, &lwork, &info);    
    return info;
}
