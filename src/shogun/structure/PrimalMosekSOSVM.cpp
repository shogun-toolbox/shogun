/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Shell Hu
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifdef USE_MOSEK

#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/List.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/structure/PrimalMosekSOSVM.h>
#include <shogun/loss/HingeLoss.h>

using namespace shogun;

CPrimalMosekSOSVM::CPrimalMosekSOSVM()
: CLinearStructuredOutputMachine(),
	po_value(0.0)
{
	init();
}

CPrimalMosekSOSVM::CPrimalMosekSOSVM(
		CStructuredModel*  model,
		CStructuredLabels* labs)
: CLinearStructuredOutputMachine(model, labs),
	po_value(0.0)
{
	init();
}

void CPrimalMosekSOSVM::init()
{
	SG_ADD(&m_slacks, "slacks", "Slacks vector", MS_NOT_AVAILABLE);
	//FIXME model selection available for SO machines
	SG_ADD(&m_regularization, "regularization", "Regularization constant", MS_NOT_AVAILABLE);
	SG_ADD(&m_epsilon, "epsilon", "Violation tolerance", MS_NOT_AVAILABLE);
	SG_ADD(&m_lb, "lb", "Lower bounds", MS_NOT_AVAILABLE);
	SG_ADD(&m_ub, "ub", "Upper bounds", MS_NOT_AVAILABLE);

	m_regularization = 1.0;
	m_epsilon = 0.0;
}

CPrimalMosekSOSVM::~CPrimalMosekSOSVM()
{
}

bool CPrimalMosekSOSVM::train_machine(CFeatures* data)
{
	SG_DEBUG("Entering CPrimalMosekSOSVM::train_machine.\n");
	if (data)
		set_features(data);

	CFeatures* model_features = get_features();
	// Initialize the model for training
	m_model->init_training();
	// Check that the scenary is correct to start with training
	m_model->check_training_setup();
	SG_DEBUG("The training setup is correct.\n");

	// Dimensionality of the joint feature space
	int32_t M = m_model->get_dim();
	// Number of auxiliary variables in the optimization vector
	int32_t num_aux = m_model->get_num_aux();
	// Number of auxiliary constraints
	int32_t num_aux_con = m_model->get_num_aux_con();
	// Number of training examples
	int32_t N = model_features->get_num_vectors();

	SG_DEBUG("M=%d, N =%d, num_aux=%d, num_aux_con=%d.\n", M, N, num_aux, num_aux_con);

	// Interface with MOSEK
	CMosek* mosek = new CMosek(0, M+num_aux+N);
	SG_REF(mosek);
	REQUIRE(mosek->get_rescode() == MSK_RES_OK, "Mosek object could not be properly created in PrimalMosekSOSVM training.\n");

	// Initialize the terms of the optimization problem
	SGMatrix< float64_t > A, B, C;
	SGVector< float64_t > a, b, lb, ub;
	m_model->init_primal_opt(m_regularization, A, a, B, b, lb, ub, C);

	REQUIRE(lb.vlen == 0 || lb.vlen == M,
		"%s::train_machine(): lb.vlen can only be 0 or w.vlen!\n", get_name());

	REQUIRE(ub.vlen == 0 || ub.vlen == M,
		"%s::train_machine(): ub.vlen can only be 0 or w.vlen!\n", get_name());

	if (lb.vlen == M)
		set_lower_bounds(lb);

	if (ub.vlen == M)
		set_upper_bounds(ub);

	SG_DEBUG("Regularization used in PrimalMosekSOSVM equal to %.2f.\n", m_regularization);

	// Input terms of the problem that do not change between iterations
	REQUIRE(mosek->init_sosvm(M, N, num_aux, num_aux_con, C, m_lb, m_ub, A, b) == MSK_RES_OK,
		"Mosek error in PrimalMosekSOSVM initializing SO-SVM.\n")

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
		CList* list = new CList(true);
		results->push_back(list);
	}

	// Initialize variables used in the loop
	int32_t     num_con     = num_aux_con;	// number of constraints
	int32_t     old_num_con = num_con;
	bool        exception   = false;
	index_t     iteration   = 0;

	SGVector< float64_t > sol(M+num_aux+N);
	sol.zero();

	SGVector< float64_t > aux(num_aux);

	do
	{
		SG_DEBUG("Iteration #%d: Cutting plane training with num_con=%d and old_num_con=%d.\n",
				iteration, num_con, old_num_con);

		old_num_con = num_con;

		for ( int32_t i = 0 ; i < N ; ++i )
		{
			// Predict the result of the ith training example (loss-aug)
			CResultSet* result = m_model->argmax(m_w, i);

			// Compute the loss associated with the prediction (surrogate loss, max(0, \tilde{H}))
			float64_t slack = CHingeLoss().loss( compute_loss_arg(result) );
			CList* cur_list = (CList*) results->get_element(i);

			// Update the list of constraints
			if ( cur_list->get_num_elements() > 0 )
			{
				// Find the maximum loss within the elements of
				// the list of constraints
				CResultSet* cur_res = (CResultSet*) cur_list->get_first_element();
				float64_t max_slack = -CMath::INFTY;

				while ( cur_res != NULL )
				{
					max_slack = CMath::max(max_slack, CHingeLoss().loss( compute_loss_arg(cur_res) ));

					SG_UNREF(cur_res);
					cur_res = (CResultSet*) cur_list->get_next_element();
				}

				if ( slack > max_slack + m_epsilon )
				{
					// The current training example is a
					// violated constraint
					if ( ! insert_result(cur_list, result) )
					{
						exception = true;
						break;
					}

					add_constraint(mosek, result, num_con, i);
					++num_con;
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

				add_constraint(mosek, result, num_con, i);
				++num_con;
			}

			SG_UNREF(cur_list);
			SG_UNREF(result);
		}

		// Solve the QP
		SG_DEBUG("Entering Mosek QP solver.\n");

		mosek->optimize(sol);
		for ( int32_t i = 0 ; i < M+num_aux+N ; ++i )
		{
			if ( i < M )
				m_w[i] = sol[i];
			else if ( i < M+num_aux )
				aux[i-M] = sol[i];
			else
				m_slacks[i-M-num_aux] = sol[i];
		}

		SG_DEBUG("QP solved. The primal objective value is %.4f.\n", mosek->get_primal_objective_value());

		++iteration;

	} while ( old_num_con != num_con && ! exception );

	po_value = mosek->get_primal_objective_value();

	// Free resources
	SG_UNREF(results);
	SG_UNREF(mosek);
	SG_UNREF(model_features);
	return true;
}

float64_t CPrimalMosekSOSVM::compute_loss_arg(CResultSet* result) const
{
	// Dimensionality of the joint feature space
	int32_t M = m_w.vlen;

	if(result->psi_computed)
	{
		return linalg::dot(m_w, result->psi_pred) +
			result->delta -
			linalg::dot(m_w, result->psi_truth);
	}
	else if(result->psi_computed_sparse)
	{
		return result->psi_pred_sparse.dense_dot(1.0, m_w.vector, m_w.vlen, 0) +
			result->delta -
			result->psi_truth_sparse.dense_dot(1.0, m_w.vector, m_w.vlen, 0);
	}
	else
	{
		SG_ERROR("model(%s) should have either of psi_computed or psi_computed_sparse"
				"to be set true\n", m_model->get_name());
		return 0;
	}
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

	if (result->psi_computed)
	{
		for ( int i = 0 ; i < M ; ++i )
			dPsi[i] = result->psi_pred[i] - result->psi_truth[i]; // -dPsi(y)
	}
	else if(result->psi_computed_sparse)
	{
		dPsi.zero();
		result->psi_pred_sparse.add_to_dense(1.0, dPsi.vector, dPsi.vlen);
		result->psi_truth_sparse.add_to_dense(-1.0, dPsi.vector, dPsi.vlen);
	}
	else
	{
		SG_ERROR("model(%s) should have either of psi_computed or psi_computed_sparse"
				"to be set true\n", m_model->get_name());
	}

	return ( mosek->add_constraint_sosvm(dPsi, con_idx, train_idx,
			m_model->get_num_aux(), -result->delta) == MSK_RES_OK );
}


float64_t CPrimalMosekSOSVM::compute_primal_objective() const
{
	return po_value;
}

EMachineType CPrimalMosekSOSVM::get_classifier_type()
{
	return CT_PRIMALMOSEKSOSVM;
}

void CPrimalMosekSOSVM::set_regularization(float64_t C)
{
	m_regularization = C;
}

void CPrimalMosekSOSVM::set_epsilon(float64_t epsilon)
{
	m_epsilon = epsilon;
}

void CPrimalMosekSOSVM::set_lower_bounds(SGVector< float64_t > lb)
{
	m_lb = lb.clone();
}

void CPrimalMosekSOSVM::set_upper_bounds(SGVector< float64_t > ub)
{
	m_ub = ub.clone();
}

#endif /* USE_MOSEK */
