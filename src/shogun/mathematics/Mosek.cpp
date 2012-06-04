/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifdef USE_MOSEK

#include <shogun/mathematics/Mosek.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;

//TODO
CMosek::CMosek()
: CSGObject()
{
}

//TODO
CMosek::~CMosek()
{
}

void MSKAPI CMosek::print(void* handle, char str[])
{
	SG_SPRINT("%s", str);
}

MSKrescodee CMosek::wrapper_putaveclist(
		MSKtask_t & task, 
		SGMatrix< float64_t > A, int32_t nnza)
{
	// Shorthands for A dimensions
	index_t N = A.num_rows;
	index_t M = A.num_cols;

	// Indices to the rows of A to replace, all the rows
	SGVector< index_t > sub(N);
	for ( index_t i = 0 ; i < N ; ++i )
		sub[i] = i;

	// Non-zero elements of A
	SGVector< float64_t > aval(nnza);
	// For each of the rows, indices to non-zero elements
	SGVector< index_t > asub(nnza);
	// For each row, pointer to the first non-zero element
	// in aval
	SGVector< int32_t > ptrb(N);
	// Next position to write in aval and asub
	index_t idx = 0;
	// Switch if the first non-zero element in each row 
	// has been found
	bool first_nnz_found = false;

	for ( index_t i = 0 ; i < N ; ++i )
	{
		first_nnz_found = false;

		for ( index_t j = 0 ; j < M ; ++j )
		{
			if ( A[j + i*M] )
			{
				aval[idx] = A[j + i*M];
				asub[idx] = j;

				if ( !first_nnz_found )
				{
					ptrb[i] = idx;
					first_nnz_found = true;
				}

				++idx;
			}
		}

		// Handle rows whose elements are all zero
		if ( !first_nnz_found )
			ptrb[i] = ( i ? ptrb[i-1] : 0 );
	}

	// For each row, pointer to the last+1 non-zero element 
	// in aval
	SGVector< int32_t > ptre(N);
	for ( index_t i = 0 ; i < N-1 ; ++i )
		ptre[i] = ptrb[i+1];

	ptre[N-1] = nnza;

	return MSK_putaveclist(task, MSK_ACC_CON, N, sub.vector,
			ptrb.vector, ptre.vector,
			asub.vector, aval.vector);
}

MSKrescodee CMosek::wrapper_putqobj(MSKtask_t & task, SGMatrix< float64_t > Q0)
{
	// Shorthands for the dimensions of the matrix
	index_t N = Q0.num_rows;
	index_t M = Q0.num_cols;

	// Count the number of non-zero elements in the lower 
	// triangular part of the matrix
	int32_t nnz = 0;
	for ( index_t i = 0 ; i < N ; ++i )
		for ( index_t j = i ; j < M ; ++j )
			nnz += ( Q0[j + i*M] ? 1 : 0 );

	// i subscript for non-zero elements of Q0
	SGVector< index_t > qosubi(nnz);
	// j subscript for non-zero elements of Q0
	SGVector< index_t > qosubj(nnz);
	// Values of non-zero elements of Q0
	SGVector< float64_t > qoval(nnz);
	// Next position to write in the vectors
	index_t idx = 0;

	for ( index_t i = 0 ; i < N ; ++i )
		for ( index_t j = i ; j < M ; ++j )
		{
			if ( Q0[j + i*M] )
			{
				qosubi[idx] = i;
				qosubj[idx] = j;
				 qoval[idx] = Q0[j + i*M]; 

				++idx;
			}
		}

	return MSK_putqobj(task, nnz, qosubi.vector, 
			qosubj.vector, qoval.vector);
}

#endif /* USE_MOSEK */
