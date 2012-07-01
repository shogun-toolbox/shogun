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
#include <shogun/structure/PrimalMosekSOSVM.h>

using namespace shogun;

CPrimalMosekSOSVM::CPrimalMosekSOSVM()
: CLinearStructuredOutputMachine()
{
}

CPrimalMosekSOSVM::CPrimalMosekSOSVM(
		CStructuredModel*  model,
		CLossFunction*     loss,
		CStructuredLabels* labs,
		CDotFeatures*      features)
: CLinearStructuredOutputMachine(model, loss, labs, features)
{
}

void CPrimalMosekSOSVM::init()
{
	SG_ADD(&m_slacks, "m_slacks", "Slacks vector", MS_NOT_AVAILABLE);
}

CPrimalMosekSOSVM::~CPrimalMosekSOSVM()
{
}

bool CPrimalMosekSOSVM::train_machine(CFeatures* data)
{
	// Check that the scenary is correct to start with training
	m_model->check_training_setup();

	// Dimensionality of the joint feature space
	int32_t M = m_model->get_dim();
	// Number of training examples
	int32_t N = m_features->get_num_vectors();

	// Interface with MOSEK
	CMosek* mosek = new CMosek(0, M+N);
	SG_REF(mosek);
	if ( mosek->get_rescode() != MSK_RES_OK )
	{
		SG_PRINT("Mosek object could not be properly created..."
			 "aborting training of PrimalMosekSOSVM\n");

		return false;
	}

	// Initialize the terms of the optimization problem
	SGMatrix< float64_t > A, B, C;
	SGVector< float64_t > a, b, lb, ub;
	//TODO take into acount these possible constraints
	m_model->init_opt(A, a, B, b, lb, ub, C);

	// Input terms of the problem that do not change between iterations
	if ( mosek->init_sosvm(M, N, C, lb, ub) != MSK_RES_OK )
	{
		SG_PRINT("Problem occurred writing constraints in MOSEK object..."
			 "aborting training of PrimalMosekSOSVM\n");

		return false;
	}

	// Initialize the weight vector
	m_w = SGVector< float64_t >(M);
	m_w.zero();

	m_slacks = SGVector< float64_t >(N);
	m_slacks.zero();

	// Initialize the list of constraints
	// Each element in results is a list of CResultSet with the constraints 
	// associated to each training example
	CDynamicObjectArray* results = new CDynamicObjectArray(N);
	SG_REF(results);
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

	SGVector< float64_t > sol(M+N);
	sol.zero();

	do 
	{
		old_numcon = numcon;

		for ( int32_t i = 0 ; i < N ; ++i )
		{
			// Predict the result of the ith training example
			result = m_model->argmax(m_w, i);

			// Compute the loss associated with the prediction
			slack = m_loss->loss( compute_loss_arg(result) );
			cur_list = (CList*) results->get_element(i);

			// Update the list of constraints
			if ( cur_list->get_num_elements() > 0 )
			{
				// Find the maximum loss within the elements of
				// the list of constraints
				cur_res = (CResultSet*) cur_list->get_first_element();
				max_slack = -CMath::INFTY;

				while ( cur_res != NULL )
				{
					max_slack = CMath::max(max_slack,
							m_loss->loss( compute_loss_arg(cur_res) ));

					cur_res = (CResultSet*) cur_list->get_next_element();
				}

				if ( slack > max_slack )
				{
					// The current training example is a
					// violated constraint
					if ( ! insert_result(cur_list, result) )
					{
						exception = true;
						break;
					}

					add_constraint(mosek, result, numcon, i);
					++numcon;
				}
			}
			else
			{
				// First iteration of do ... while, add constraint
				if ( ! insert_result(cur_list, result) )
				{
					exception = true;
					break;
				}

				add_constraint(mosek, result, numcon, i);
				++numcon;
			}

			SG_UNREF(result);
		}

		// Solve the QP
		mosek->optimize(sol);
		for ( int32_t i = 0 ; i < M+N ; ++i )
		{
			if ( i < M )
				m_w[i] = sol[i];
			else
				m_slacks[i-M] = sol[i];
		}

	} while ( old_numcon != numcon && ! exception);

	// Free resources
	SG_UNREF(mosek);
	SG_UNREF(results);

	return true;
}

float64_t CPrimalMosekSOSVM::compute_loss_arg(CResultSet* result) const
{
	// Dimensionality of the joint feature space
	int32_t M = m_w.vlen;

	return 	SGVector< float64_t >::dot(m_w.vector, result->psi_pred.vector, M) +
		result->delta -
		SGVector< float64_t >::dot(m_w.vector, result->psi_truth.vector, M);
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

bool CPrimalMosekSOSVM::add_constraint(
		CMosek* mosek,
		CResultSet* result,
		index_t con_idx,
		index_t train_idx) const
{
	int32_t M = m_model->get_dim();
	SGVector< float64_t > dPsi(M);

	for ( int i = 0 ; i < M ; ++i )
		dPsi[i] = result->psi_pred[i] - result->psi_truth[i];

	return ( mosek->add_constraint_sosvm(dPsi, con_idx, train_idx, 
			-result->delta) == MSK_RES_OK );
}

#endif /* USE_MOSEK */
