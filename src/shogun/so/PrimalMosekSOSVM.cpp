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

#include <shogun/mathematics/Math.h>
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
	CList* result_list = new CList(true);

	// Initialize variables used in the loop
	int32_t     old_length = 0;
	float64_t   slack      = 0.0;
	float64_t   max_slack  = 0.0;
	CResultSet* result     = NULL;
	CResultSet* cur_res    = NULL;

	do 
	{
		old_length = result_list->get_num_elements();

		for ( int32_t i = 0 ; i < m_features->get_num_vectors() ; ++i )
		{
			// Predict the result of the ith training example
			result = m_model->argmax(m_w, i);

			// Compute the loss associated with the prediction
			slack = m_loss->loss( compute_loss_arg(result) );

			// Update the list of constraints
			if ( result_list->get_num_elements() > 0 )
			{
				// Find the maximum loss within the elements of the list
				// of constraints
				cur_res = (CResultSet*) result_list->get_first_element();
				max_slack = m_loss->loss( compute_loss_arg(cur_res) );

				cur_res = (CResultSet*) result_list->get_next_element();
				while ( cur_res != NULL )
				{
					max_slack = CMath::max(max_slack, 
							m_loss->loss( compute_loss_arg(cur_res) ));

					cur_res = (CResultSet*) result_list->get_next_element();
				}

				if ( slack > max_slack )
				{
					if ( ! insert_result(result_list, result) )
						return false;
				}
			}
			else
			{
					if ( ! insert_result(result_list, result) )
						return false;
			}

			// Solve the QP

		}

	} while ( old_length != result_list->get_num_elements() );

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
		SG_PRINT("Element could not be inserted in the results list..."
			 "aborting training of PrimalMosekSOSVM\n");
	}

	return succeed;
}

bool CPrimalMosekSOSVM::solve_qd(
		SGVector< float64_t > lb, 
		SGVector< float64_t > ub) const
{
	// Dimensionality of the joint feature space
	int32_t M = m_w.vlen;

	// Create mosek environment
	MSKenv_t env;
	MSKrescodee r = MSK_makeenv(&env, NULL, NULL, NULL, NULL);
	// Abort if the return code is not ok
	if ( r != MSK_RES_OK )
	{
		SG_PRINT("MOSEK environment could not be created properly... "
			 "aborting training of PrimalMosekSOSVM\n");
		return false;
	}

	// Initialize the environment
	r = MSK_initenv(env);
	// Abort if the return code is not ok
	if ( r != MSK_RES_OK )
	{
		SG_PRINT("MOSEK environment could be be initialized properly... "
			 "aborting training of PrimalMosekSOSVM\n");
	}

	// Create the optimization task
	MSKtask_t task;
	//TODO store numcon and numvar
	r = MSK_maketask(env, NUMCON, NUMVAR, &task);
	// Abort if the return code is not ok
	if ( r != MSK_RES_OK )
	{
		SG_PRINT("MOSEK task could be created properly... "
			 "aborting training of PrimalMosekSOSVM\n");
	}

	// Give an estimate of the size of the input data to increase the speed of
	// inputting
	// TODO
	
	// Append empty constraints initialized with no bounds
	r = MSK_append(task, MSK_ACC_CON, NUMCON);

	// Append optimization variables initialized to zero
	r = MSK_append(task, MSK_ACC_VAR, NUMVAR);

	// Set the constant term in the objective equal to zero
	r = MSK_putcfix(task, 0.0);

	for ( int32_t j = 0 ; j < NUMVAR && r == MSK_RES_OK ; ++j )
	{
		// Set the linear term c_j in the objective
		if ( j < M )
			r = MSK_putcj(task, j, 0.0);
		else
			r = MSK_putcj(task, j, 1.0);

		// Set the bounds on x_j: blx[j] <= x_j <= bux[j]
		// TODO assumption taken that x_j is ranged, i.e. blx[j] and 
		// bux[j] are always finite
		if ( j < M )
		{
			r = MSK_putbound(task, MSK_ACC_VAR, j, MSK_BK_RA, 
					lb[j], ub[j]);
		}
	}

	for ( int32_t i = 0 ; i < NUMCON && r == MSK_RES_OK ; ++i )
	{
		// Input row i of A
		if ( r == MSK_RES_OK )
		{
			SGVector< int32_t > nzi_idxs;
			SGVector< int32_t > nzi_values;
			int32_t nzi;

			design_A_row(i, nzi, nzi_idxs, nzi_values);

			r = MSK_putavec(task, MSK_ACC_CON, i, nzi, 
					nzi_idxs.vector, nzi_values.vector);
		}

		// Set the bounds on constraints (b in Ax <= b)
		if ( r == MSK_RES_OK )
		{
			r = MSK_putbound(task, MSK_ACC_CON, i, MSK_BK_UP,
					-MSK_INFINITY, 
					-predicted_delta_loss(i) );
		}
	}

	// Input matrix Q
	if ( r == MSK_RES_OK )
	{
	
	}

	// Run optimizer

}	

//TODO
void CPrimalMosekSOSVM::design_A_row(
		int32_t  		row_idx,
		int32_t& 		nzi,
		SGVector< int32_t >	nzi_idxs,
		SGVector< int32_t >	nzi_values) const
{
}

//TODO
float64_t CPrimalMosekSOSVM::predicted_delta_loss(int32_t idx)
{
	// Predict yhat_i using the current weight vector
	// Compute the delta loss of yhat_i using the model
	return 0.0;
}

#endif /* USE_MOSEK */
