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

#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/List.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Mosek.h>
#include <shogun/so/PrimalMosekSOSVM.h>

using namespace shogun;

CPrimalMosekSOSVM::CPrimalMosekSOSVM()
: CLinearStructuredOutputMachine()
{
}

CPrimalMosekSOSVM::CPrimalMosekSOSVM(
		CStructuredModel*  model,
		CLossFunction*     loss,
		CStructuredLabels* labs,
		CFeatures*         features)
: CLinearStructuredOutputMachine(model, loss, labs, features)
{
}

CPrimalMosekSOSVM::~CPrimalMosekSOSVM()
{
}

bool CPrimalMosekSOSVM::train_machine(CFeatures* data)
{
	// Dimensionality of the joint feature space
	int32_t M = m_model->get_dim();
	// Number of training examples
	int32_t N = m_features->get_num_vectors();

	// Initialize the terms of the optimization problem
	SGMatrix< float64_t > A, B, C;
	SGVector< float64_t > a, b, lb, ub;
	m_model->init_opt(A, a, B, b, lb, ub, C);

	// Assume diagonal regularization matrix with just one value
	// float64_t lambda = C(0, 0);

	// Initialize the weight vector
	m_w = SGVector< float64_t >(M);
	m_w.zero();

	// Initialize the list of constraints
	// Each element in results is a list of CResultSet with the 
	// constraints associated to each training example
	CDynamicObjectArray* results = new CDynamicObjectArray(N);
	for ( int32_t i = 0 ; i < N ; ++i )
	{
		results->push_back( new CList(true) );
	}

	// Initialize variables used in the loop
	int32_t     numcon     = 0;	// number of constraints
	int32_t     old_numcon = 0;
	float64_t   slack      = 0.0;
	float64_t   max_slack  = 0.0;
	CResultSet* result     = NULL;
	CResultSet* cur_res    = NULL;
	CList*      cur_list   = NULL;
	bool        exception  = false;

	do 
	{
		old_numcon = numcon;

		for ( int32_t i = 0 ; i < N ; ++i )
		{
			// Predict the result of the ith training example
			result = m_model->get_argmax(m_w, i);

			// Compute the loss associated with the prediction
			slack = m_loss->loss( compute_loss_arg(result) );

			cur_list = (CList*) results->get_element(i);

			// Update the list of constraints
			if ( cur_list->get_num_elements() > 0 )
			{
				// Find the maximum loss within the elements of the list
				// of constraints
				cur_res = (CResultSet*) cur_list->get_first_element();
				max_slack = m_loss->loss( compute_loss_arg(cur_res) );

				while ( cur_res != NULL )
				{
					cur_res = (CResultSet*) cur_list->get_next_element();

					max_slack = CMath::max(max_slack, 
							m_loss->loss( compute_loss_arg(cur_res) ));
				}

				if ( slack > max_slack )
				{
					if ( ! insert_result(cur_list, result) )
					{
						exception = true;
						break;
					}

					++numcon;
				}
			}
			else
			{
				if ( ! insert_result(cur_list, result) )
				{
					exception = true;
					break;
				}

				++numcon;
			}

			// Solve the QP
			solve_qp(A, C, lb, ub);

		}

	} while ( old_numcon != numcon && ! exception);

	// Free resources
	// TODO

	return true;
}

void CPrimalMosekSOSVM::register_parameters()
{
	SG_ADD(&m_w, "m_w", "Weight vector", MS_NOT_AVAILABLE);
}

float64_t CPrimalMosekSOSVM::compute_loss_arg(CResultSet* result) const
{
	// Dimensionality of the joint feature space
	int32_t M = m_w.vlen;

	return 	CMath::dot(m_w.vector, result->psi_pred.vector, M) +
		result->delta -
		CMath::dot(m_w.vector, result->psi_truth.vector, M);
}

bool CPrimalMosekSOSVM::insert_result(CList* result_list, CResultSet* result) const
{
	bool succeed = result_list->insert_element(result);

	if ( ! succeed )
	{
		SG_PRINT("ResultSet could not be inserted in the list..."
			 "aborting training of PrimalMosekSOSVM\n");
	}

	return succeed;
}

bool CPrimalMosekSOSVM::solve_qp(
		SGMatrix< float64_t > A,
		SGMatrix< float64_t > C,
		SGVector< float64_t > lb, 
		SGVector< float64_t > ub) const
{
	// Return code (true = SUCCESS, false = FAILURE)
	bool ret = true;

	// Number of variables, constraints
	int32_t num_var = m_model->get_dim() + m_features->get_num_vectors();
	int32_t num_con = A.num_rows;

	// Count number of non-zero elements in A
	int32_t nnza = CMath::get_num_nonzero(A.matrix, A.num_rows*A.num_cols);

	// Dimensionality of the joint feature space
	int32_t M = m_w.vlen;

	// Create mosek environment
	MSKenv_t env;
	MSKrescodee r = MSK_makeenv(&env, NULL, NULL, NULL, NULL);

	// Direct the environment's log stream to SG_PRINT
	if ( r == MSK_RES_OK )
	{
		r = MSK_linkfunctoenvstream(env, MSK_STREAM_LOG, 
				NULL, CMosek::print);
	}

	// Initialize the environment
	r = MSK_initenv(env);

	// Create the optimization task
	MSKtask_t task;
	if ( r == MSK_RES_OK )
	{
		r = MSK_maketask(env, num_con, num_var, &task);
	}

	// Direct the task's log stream to SG_PRINT
	if ( r == MSK_RES_OK )
	{
		r = MSK_linkfunctotaskstream(task, MSK_STREAM_LOG,
				NULL, CMosek::print);
	}

	if ( r == MSK_RES_OK )
	{
		// Give an estimate of the size of the input data 
		// to increase the speed of inputting
		r = MSK_putmaxnumvar(task, num_var);
		r = MSK_putmaxnumcon(task, num_con);
		r = MSK_putmaxnumanz(task, nnza);

		// Append empty constraints initialized with no bounds
		r = MSK_append(task, MSK_ACC_CON, num_con);

		// Append optimization variables initialized to zero
		r = MSK_append(task, MSK_ACC_VAR, num_var);

		// Set the constant term in the objective equal to zero
		r = MSK_putcfix(task, 0.0);
	}

	if ( r == MSK_RES_OK )
	{
		for ( int32_t j = 0 ; j < num_var && r == MSK_RES_OK ; ++j )
		{
			// Set the linear term c_j in the objective
			if ( j < M )
				r = MSK_putcj(task, j, 0.0);
			else
				r = MSK_putcj(task, j, 1.0);

			// Set the bounds on x_j: blx[j] <= x_j <= bux[j]
			// FIXME assumption taken that x_j is ranged, i.e. blx[j] and 
			// bux[j] are always finite
			if ( j < M )
			{
				r = MSK_putbound(task, MSK_ACC_VAR, j, MSK_BK_RA, 
						lb[j], ub[j]);
			}
		}
	}

	if ( r == MSK_RES_OK )
	{
		for ( int32_t i = 0 ; i < num_con && r == MSK_RES_OK ; ++i )
		{
			// Set the bounds on constraints (b in Ax <= b)
			r = MSK_putbound(task, MSK_ACC_CON, i, MSK_BK_UP,
					-MSK_INFINITY, 
					-predicted_delta_loss(i));
		}
	}

	// Input matrix A
	if ( r == MSK_RES_OK )
	{
		r = CMosek::wrapper_putaveclist(task, A, nnza);
	}

	// Input matrix Q^0
	if ( r == MSK_RES_OK )
	{
		// FIXME C != Q0 but Q0 is just an extended version of
		// C with zeroes and zeroes make no difference in putqobj
		r = CMosek::wrapper_putqobj(task, C);
	}

	// Run optimizer
	if ( r == MSK_RES_OK )
	{
		r = MSK_optimize(task);

#ifdef DEBUG_PRIMAL_MOSEK_SOSVM
		// Print a summary containing information about the solution
		MSK_solutionsummary(task, MSK_STREAM_LOG);
#endif
	}

	// Read the solution
	if ( r == MSK_RES_OK )
	{
		// Solution status
		MSKsolstae solsta;
		// FIXME posible solutions are:
		// MSK_SOL_ITR: the interior solution
		// MSK_SOL_BAS: the basic solution
		// MSK_SOL_ITG: the integer solution
		MSK_getsolutionstatus(task, MSK_SOL_ITR, NULL, &solsta);

		switch (solsta)
		{
		case MSK_SOL_STA_OPTIMAL:
		case MSK_SOL_STA_NEAR_OPTIMAL:
			MSK_getsolutionslice(task, 
					// Request from the interior solution
					MSK_SOL_ITR,
					// the optimization vector
					MSK_SOL_ITEM_XX,
					// FIXME just interested in weight
					// vector's elements?
					0, 
					m_w.vlen, 
					m_w.vector);
			break;
#ifdef DEBUG_PRIMAL_MOSEK_SOSVM
		case MSK_SOL_STA_DUAL_INFEAS_CER:
		case MSK_SOL_STA_PRIM_INFEAS_CER:
		case MSK_SOL_STA_NEAR_DUAL_INFEAS_CER:
		case MSK_SOL_STA_NEAR_PRIM_INFEAS_CER:
			SG_PRINT("Primal or dual infeasibility certificate "
				 "found\n");
			break;
		case MSK_SOL_STA_UNKNOWN:
			SG_PRINT("Undetermined solution status\n");
			break;
		default:
			SG_PRINT("Other solution status\n");
#endif
		}
	}

	// In case any error occurred, print the appropriate error message
	if ( r != MSK_RES_OK )
	{
		char symname[MSK_MAX_STR_LEN];
		char desc[MSK_MAX_STR_LEN];

		MSK_getcodedesc(r, symname, desc);

		SG_PRINT("An error occurred optimizing with MOSEK\n");
		SG_PRINT("ERROR %s - '%s'\n", symname, desc);
	}

	// Free resources
	MSK_deletetask(&task);
	MSK_deleteenv(&env);

	return ret;
}

//TODO
float64_t CPrimalMosekSOSVM::predicted_delta_loss(int32_t idx) const
{
	// Predict yhat_i using the current weight vector
	// Compute the delta loss of yhat_i using the model
	return 0.0;
}

#endif /* USE_MOSEK */
