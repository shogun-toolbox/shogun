/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/mathematics/Math.h>
#include <shogun/so/VanillaStructuredOutputMachine.h>

using namespace shogun;

CVanillaStructuredOutputMachine::CVanillaStructuredOutputMachine()
: CLinearStructuredOutputMachine()
{
}

CVanillaStructuredOutputMachine::CVanillaStructuredOutputMachine(
		CStructuredModel* model,
		CStructuredLoss*  loss,
		CStructuredLabels* labs,
		CFeatures*         features)
: CLinearStructuredOutputMachine(model, loss, labs, features)
{
}

CVanillaStructuredOutputMachine::~CVanillaStructuredOutputMachine()
{
}

bool CVanillaStructuredOutputMachine::train_machine(CFeatures* data)
{
	// Dimensionality of the joint feature space
	int32_t N = m_model->get_dim();

	//TODO return A, a, B, b, lb, ub, C
	SGMatrix< float64_t > C(1, 1);
	m_model->init();

	// Assume diagonal regularization matrix with just one value
	float64_t lambda = C(0, 0);

	// Initialize the weight vector
	m_w = SGVector< float64_t >(N);
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

	return true;
}

void CVanillaStructuredOutputMachine::register_parameters()
{
	SG_ADD(&m_w, "m_w", "Weight vector", MS_NOT_AVAILABLE);
}

float64_t CVanillaStructuredOutputMachine::compute_loss_arg(CResultSet* result) const
{
	// Dimensionality of the joint feature space
	int32_t N = m_w.vlen;

	return 	CMath::dot(m_w.vector, result->psi_pred.vector, N) +
		result->delta -
		CMath::dot(m_w.vector, result->psi_truth.vector, N);
}

bool CVanillaStructuredOutputMachine::insert_result(CList* result_list, CResultSet* result) const
{
	bool succeed = result_list->insert_element(result);
	if ( ! succeed )
	{
		SG_PRINT("Element couldn't be inserted in the results list"
			 " aborting training of VanillaStructuredOutputMachine\n");
	}

	return succeed;
}
