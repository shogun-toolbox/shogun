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

//#define DEBUG_MOSEK
//#define DEBUG_SOLUTION

#include <mathematics/Math.h>
#include <mathematics/Mosek.h>
#include <lib/SGVector.h>

using namespace shogun;

CMosek::CMosek()
: CSGObject()
{
}

CMosek::CMosek(int32_t num_con, int32_t num_var)
: CSGObject()
{
	// Create MOSEK environment
#if (MSK_VERSION_MAJOR == 6)
	m_rescode = MSK_makeenv(&m_env, NULL, NULL, NULL, NULL);
#elif (MSK_VERSION_MAJOR == 7)
	m_rescode = MSK_makeenv(&m_env, NULL);
#else
	#error "Unsupported Mosek version"
#endif

#ifdef DEBUG_MOSEK
	// Direct the environment's log stream to SG_PRINT
	if ( m_rescode == MSK_RES_OK )
	{
		m_rescode = MSK_linkfunctoenvstream(m_env, MSK_STREAM_LOG,
				NULL, CMosek::print);
	}
#endif

	// Initialize the environment
	if ( m_rescode == MSK_RES_OK )
	{
		m_rescode = MSK_initenv(m_env);
	}

	// Create an optimization task
	if ( m_rescode == MSK_RES_OK )
	{
		m_rescode = MSK_maketask(m_env, num_con, num_var, &m_task);
	}

#ifdef DEBUG_MOSEK
	// Direct the task's log stream to SG_PRINT
	if ( m_rescode == MSK_RES_OK )
	{
		m_rescode = MSK_linkfunctotaskstream(m_task, MSK_STREAM_LOG,
				NULL, CMosek::print);
	}
#endif
}

CMosek::~CMosek()
{
	delete_problem();
}

void MSKAPI CMosek::print(void* handle, char str[])
{
	SG_SPRINT("%s", str)
}

MSKrescodee CMosek::init_sosvm(int32_t M, int32_t N,
		int32_t num_aux, int32_t num_aux_con,
		SGMatrix< float64_t > C, SGVector< float64_t > lb,
		SGVector< float64_t > ub, SGMatrix< float64_t > A,
		SGVector< float64_t > b)
{
	// Give an estimate of the size of the input data to increase the
	// speed of inputting
	int32_t num_var = M+N+num_aux;
	int32_t num_con = N*N+num_aux_con;
	// NOTE: However, to input this step is completely optional and MOSEK
	// will automatically allocate more resources if necessary
	m_rescode = MSK_putmaxnumvar(m_task, num_var);
	// Neither the number of constraints nor the number of non-zero elements
	// is known a priori, rough estimates are given here
	m_rescode = MSK_putmaxnumcon(m_task, num_con);
	// A = [-dPsi(y) | -I_N ] with M+N columns => max. M+1 nnz per row
	m_rescode = MSK_putmaxnumanz(m_task, (M+1)*N*N);

	// Append optimization variables initialized to zero
#if (MSK_VERSION_MAJOR == 6)
	m_rescode = MSK_append(m_task, MSK_ACC_VAR, num_var);
#elif (MSK_VERSION_MAJOR == 7)
	m_rescode = MSK_appendvars(m_task, num_var);
#else
	#error "Unsupported Mosek version"
#endif
	// Append empty constraints initialized with no bounds
#if (MSK_VERSION_MAJOR == 6)
	m_rescode = MSK_append(m_task, MSK_ACC_CON, num_con);
#elif (MSK_VERSION_MAJOR == 7)
	m_rescode = MSK_appendcons(m_task, num_con);
#else
	#error "Unsupported Mosek version"
#endif
	// Set the constant term in the objective equal to zero
	m_rescode = MSK_putcfix(m_task, 0.0);

	for ( int32_t j = 0 ; j < num_var && m_rescode == MSK_RES_OK ; ++j )
	{
		// Set the linear term c_j in the objective
		if ( j < M+num_aux )
			m_rescode = MSK_putcj(m_task, j, 0.0);
		else
			m_rescode = MSK_putcj(m_task, j, 1.0);

		// Set the bounds on x_j: blx[j] <= x_j <= bux[j]
		// TODO set bounds lb and ub given by init_opt for w
		if ( j < M )
		{
			m_rescode = MSK_putbound(m_task, MSK_ACC_VAR, j,
					MSK_BK_FR, -MSK_INFINITY, +MSK_INFINITY);
		}

		// The slack and the auxiliary variables are required to be positive
		if ( j >= M )
		{
			m_rescode = MSK_putbound(m_task, MSK_ACC_VAR, j,
					MSK_BK_LO, 0.0, +MSK_INFINITY);
		}
	}

	// Input the matrix Q^0 for the objective
	//
	// NOTE: In MOSEK we minimize x'*Q^0*x. C != Q0 but Q0 is
	// just an extended version of C with zeros that make no
	// difference in MOSEK's sparse representation
	m_rescode = wrapper_putqobj(C);

	// Input the matrix A and the vector b for the contraints A*x <= b
	m_rescode = wrapper_putaveclist(m_task, A);
	m_rescode = wrapper_putboundlist(m_task, b);

	REQUIRE(m_rescode == MSK_RES_OK, "MOSEK Error in CMosek::init_sosvm(). "
			"Enable DEBUG_MOSEK for details.\n");

	return m_rescode;
}

MSKrescodee CMosek::add_constraint_sosvm(
		SGVector< float64_t > dPsi,
		index_t con_idx,
		index_t train_idx,
		int32_t num_aux,
		float64_t bi)
{
	// Count the number of non-zero element in dPsi
	int32_t nnz = CMath::get_num_nonzero(dPsi.vector, dPsi.vlen);
	// Indices of the non-zero elements in the row of A to add
	SGVector< index_t > asub(nnz+1); // +1 because of the -1 element
	// Values of the non-zero elements
	SGVector< float64_t > aval(nnz+1);
	// Next element to add in asub and aval
	index_t idx = 0;

	for ( int32_t i = 0 ; i < dPsi.vlen ; ++i )
	{
		if ( dPsi[i] != 0 )
		{
			asub[idx] = i;
			aval[idx] = dPsi[i];
			++idx;
		}
	}

	ASSERT(idx == nnz)

	asub[idx] = dPsi.vlen + num_aux + train_idx;
	aval[idx] = -1;

#if (MSK_VERSION_MAJOR == 6)
	m_rescode = MSK_putavec(m_task, MSK_ACC_CON, con_idx, nnz+1,
			asub.vector, aval.vector);
#elif (MSK_VERSION_MAJOR == 7)
	m_rescode = MSK_putarow(m_task, con_idx, nnz+1, asub.vector, aval.vector);
#else
	#error "Unsupported Mosek version"
#endif

	if ( m_rescode == MSK_RES_OK )
	{
		m_rescode = MSK_putbound(m_task, MSK_ACC_CON, con_idx,
				MSK_BK_UP, -MSK_INFINITY, bi);
	}

	return m_rescode;
}

MSKrescodee CMosek::wrapper_putaveclist(
		MSKtask_t & task,
		SGMatrix< float64_t > A)
{
	// Indices to the rows of A to replace, all the rows
	SGVector< index_t > sub(A.num_rows);
	for ( index_t i = 0 ; i < A.num_rows ; ++i )
		sub[i] = i;

	// Non-zero elements of A
	int32_t nnza = CMath::get_num_nonzero(A.matrix, A.num_rows*A.num_cols);
	SGVector< float64_t > aval(nnza);
	// For each of the rows, indices to non-zero elements
	SGVector< index_t > asub(nnza);
	// For each row, pointer to the first non-zero element
	// in aval
	SGVector< int32_t > ptrb(A.num_rows);
	// Next position to write in aval and asub
	index_t idx = 0;
	// Switch if the first non-zero element in each row
	// has been found
	bool first_nnz_found = false;

	for ( index_t i = 0 ; i < A.num_rows ; ++i )
	{
		first_nnz_found = false;

		for ( index_t j = 0 ; j < A.num_cols ; ++j )
		{
			if ( A(i,j) )
			{
				aval[idx] = A(i,j);
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
		// TODO does it make sense that a row in A has all its elements
		// equal to zero?
		if ( !first_nnz_found )
			ptrb[i] = ( i ? ptrb[i-1] : 0 );
	}

	// For each row, pointer to the last+1 non-zero element
	// in aval
	SGVector< int32_t > ptre(A.num_rows);
	for ( index_t i = 0 ; i < A.num_rows-1 ; ++i )
		ptre[i] = ptrb[i+1];

	if ( A.num_rows > 0 )
		ptre[A.num_rows-1] = nnza;

	MSKrescodee ret;
#if (MSK_VERSION_MAJOR == 6)
	ret = MSK_putaveclist(task, MSK_ACC_CON, A.num_rows, sub.vector,
			ptrb.vector, ptre.vector,
			asub.vector, aval.vector);
#elif (MSK_VERSION_MAJOR == 7)
	ret = MSK_putarowlist(task, A.num_rows, sub.vector, ptrb.vector, ptre.vector,
			asub.vector, aval.vector);
#else
	#error "Unsupported Mosek version"
#endif

	REQUIRE(ret == MSK_RES_OK, "MOSEK Error in CMosek::wrapper_putaveclist(). "
			"Enable DEBUG_MOSEK for details.\n");

	return ret;
}

MSKrescodee CMosek::wrapper_putboundlist(MSKtask_t & task, SGVector< float64_t > b)
{
	// Indices to the bounds that should be replaced, b.vlen bounds starting
	// from zero
	SGVector< index_t > sub(b.vlen);
	for ( index_t i = 0 ; i < b.vlen ; ++i )
		sub[i] = i;

	// Type of the bounds and lower bound values
	MSKboundkeye* bk = SG_MALLOC(MSKboundkeye, b.vlen);
	SGVector< float64_t > bl(b.vlen);
	for ( index_t i = 0 ; i < b.vlen ; ++i )
	{
		bk[i] =  MSK_BK_UP;
		bl[i] = -MSK_INFINITY;
	}

	MSKrescodee ret =  MSK_putboundlist(task, MSK_ACC_CON, b.vlen, sub.vector,
			bk, bl.vector, b.vector);

	SG_FREE(bk);

	REQUIRE(ret == MSK_RES_OK, "MOSEK Error in CMosek::wrapper_putboundlist(). "
			"Enable DEBUG_MOSEK for details.\n");

	return ret;
}

MSKrescodee CMosek::wrapper_putqobj(SGMatrix< float64_t > Q0) const
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

	return MSK_putqobj(m_task, nnz, qosubi.vector,
			qosubj.vector, qoval.vector);
}

MSKrescodee CMosek::optimize(SGVector< float64_t > sol)
{
	m_rescode = MSK_optimize(m_task);

#ifdef DEBUG_MOSEK
	// Print a summary containing information about the solution
	MSK_solutionsummary(m_task, MSK_STREAM_LOG);
#endif

	// Read the solution
	if ( m_rescode == MSK_RES_OK )
	{
		// Solution status
		MSKsolstae solsta;
		// FIXME posible solutions are:
		// MSK_SOL_ITR: the interior solution
		// MSK_SOL_BAS: the basic solution
		// MSK_SOL_ITG: the integer solution
#if (MSK_VERSION_MAJOR == 6)
		MSK_getsolutionstatus(m_task, MSK_SOL_ITR, NULL, &solsta);
#elif (MSK_VERSION_MAJOR == 7)
		MSK_getsolsta(m_task, MSK_SOL_ITR, &solsta);
#else
	#error "Unsupported Mosek Version"
#endif
		switch (solsta)
		{
		case MSK_SOL_STA_OPTIMAL:
		case MSK_SOL_STA_NEAR_OPTIMAL:
			MSK_getsolutionslice(m_task,
					// Request the interior solution
					MSK_SOL_ITR,
					// of the optimization vector
					MSK_SOL_ITEM_XX,
					0,
					sol.vlen,
					sol.vector);
#ifdef DEBUG_SOLUTION
			sol.display_vector("Solution");
#endif
			break;
		case MSK_SOL_STA_DUAL_INFEAS_CER:
		case MSK_SOL_STA_PRIM_INFEAS_CER:
		case MSK_SOL_STA_NEAR_DUAL_INFEAS_CER:
		case MSK_SOL_STA_NEAR_PRIM_INFEAS_CER:
#ifdef DEBUG_MOSEK
			SG_PRINT("Primal or dual infeasibility "
				 "certificate found\n");
#endif
			break;
		case MSK_SOL_STA_UNKNOWN:
#ifdef DEBUG_MOSEK
			SG_PRINT("Undetermined solution status\n")
#endif
			break;
		default:
#ifdef DEBUG_MOSEK
			SG_PRINT("Other solution status\n")
#endif
			break;	// to avoid compile error when DEBUG_MOSEK
				// is not defined
		}
	}

	// In case any error occurred, print the appropriate error message
	if ( m_rescode != MSK_RES_OK )
	{
		char symname[MSK_MAX_STR_LEN];
		char desc[MSK_MAX_STR_LEN];

		MSK_getcodedesc(m_rescode, symname, desc);

		SG_PRINT("An error occurred optimizing with MOSEK\n")
		SG_PRINT("ERROR %s - '%s'\n", symname, desc)
	}

	return m_rescode;
}

void CMosek::delete_problem()
{
	MSK_deletetask(&m_task);
	MSK_deleteenv(&m_env);
}

void CMosek::display_problem()
{
	int32_t num_var, num_con;
	m_rescode = MSK_getnumvar(m_task, &num_var);
	m_rescode = MSK_getnumcon(m_task, &num_con);

	SG_PRINT("\nMatrix Q^0:\n")
	for ( int32_t i = 0 ; i < num_var ; ++i )
	{
		for ( int32_t j = 0 ; j < num_var ; ++j )
		{
			float64_t qij;
			m_rescode = MSK_getqobjij(m_task, i, j, &qij);
			if ( qij != 0.0 )
				SG_PRINT("(%d,%d)\t%.2f\n", i, j, qij)
		}
	}
	SG_PRINT("\n")

	SG_PRINT("\nVector c:\n")
	SGVector< float64_t > c(num_var);
	m_rescode = MSK_getc(m_task, c.vector);
	c.display_vector();

	SG_PRINT("\n\nMatrix A:\n")
	for ( int32_t i = 0 ; i < num_con ; ++i )
	{
		for ( int32_t j = 0 ; j < num_var ; ++j )
		{
			float64_t aij;
			m_rescode = MSK_getaij(m_task, i, j, &aij);
			if ( aij != 0.0 )
				SG_PRINT("(%d,%d)\t%.2f\n", i, j, aij)
		}
	}
	SG_PRINT("\n")

	SG_PRINT("\nConstraint Bounds, vector b:\n")
	for ( int32_t i = 0 ; i < num_con ; ++i )
	{
		MSKboundkeye bk;
		float64_t bl, bu;
		m_rescode = MSK_getbound(m_task, MSK_ACC_CON, i, &bk, &bl, &bu);

		SG_PRINT("%6.2f %6.2f\n", bl, bu)
	}

	SG_PRINT("\nVariable Bounds, vectors lb and ub:\n")
	for ( int32_t i = 0 ; i < num_var ; ++i )
	{
		MSKboundkeye bk;
		float64_t bl, bu;
		m_rescode = MSK_getbound(m_task, MSK_ACC_VAR, i, &bk, &bl, &bu);

		SG_PRINT("%6.2f %6.2f\n", bl, bu)
	}
	SG_PRINT("\n")
}


float64_t CMosek::get_primal_objective_value() const
{
	float64_t po = 0.0;
	MSK_getprimalobj(m_task, MSK_SOL_ITR, &po);

	return po;
}

#endif /* USE_MOSEK */
