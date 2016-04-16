/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Shell Hu
 * Copyright (C) 2014 Shell Hu
 */

#include <shogun/mathematics/Math.h>
#include <shogun/structure/FWSOSVM.h>
#include <shogun/labels/LabelsFactory.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/linalg.h>

using namespace shogun;
using namespace linalg;


CFWSOSVM::CFWSOSVM()
: CLinearStructuredOutputMachine()
{
	init();
}

CFWSOSVM::CFWSOSVM(
		CStructuredModel*  model,
		CStructuredLabels* labs,
		bool do_line_search,
		bool verbose)
: CLinearStructuredOutputMachine(model, labs)
{
	REQUIRE(model != NULL && labs != NULL,
		"%s::CFWSOSVM(): model and labels cannot be NULL!\n", get_name());

	REQUIRE(labs->get_num_labels() > 0,
		"%s::CFWSOSVM(): number of labels should be greater than 0!\n", get_name());

	init();
	m_lambda = 1.0 / labs->get_num_labels();
	m_do_line_search = do_line_search;
	m_verbose = verbose;
}

void CFWSOSVM::init()
{
	SG_ADD(&m_lambda, "lambda", "Regularization constant", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_iter, "num_iter", "Number of iterations", MS_NOT_AVAILABLE);
	SG_ADD(&m_do_line_search, "do_line_search", "Do line search", MS_NOT_AVAILABLE);
	SG_ADD(&m_gap_threshold, "gap_threshold", "Gap threshold", MS_NOT_AVAILABLE);
	SG_ADD(&m_ell, "ell", "Average loss", MS_NOT_AVAILABLE);

	m_lambda = 1.0;
	m_num_iter = 50;
	m_do_line_search = true;
	m_gap_threshold = 0.1;
	m_ell = 0;
}

CFWSOSVM::~CFWSOSVM()
{
}

EMachineType CFWSOSVM::get_classifier_type()
{
	return CT_FWSOSVM;
}

bool CFWSOSVM::train_machine(CFeatures* data)
{
	SG_DEBUG("Entering CFWSOSVM::train_machine.\n");
	if (data)
		set_features(data);

	// Initialize the model for training
	m_model->init_training();
	// Check that the scenary is correct to start with training
	m_model->check_training_setup();
	SG_DEBUG("The training setup is correct.\n");

	// Dimensionality of the joint feature space
	int32_t M = m_model->get_dim();
	// Number of training examples
	int32_t N = CLabelsFactory::to_structured(m_labels)->get_num_labels();

	SG_DEBUG("M=%d, N =%d.\n", M, N);

	// Initialize the weight vector
	m_w = SGVector<float64_t>(M);
	m_w.zero();

	// Initialize the average loss
	m_ell = 0;

	// logging
	if (m_verbose)
	{
		if (m_helper != NULL)
			SG_UNREF(m_helper);

		m_helper = new CSOSVMHelper();
		SG_REF(m_helper);
	}

	// Main loop
	int32_t k = 0;
	SGVector<float64_t> w_s(M);
	float64_t ell_s = 0;
	for (int32_t pi = 0; pi < m_num_iter; ++pi)
	{
		// init w_s and ell_s
		k = pi;
		w_s.zero();
		ell_s = 0;

		for (int32_t si = 0; si < N; ++si)
		{
			// 1) solve the loss-augmented inference for point si
			CResultSet* result = m_model->argmax(m_w, si);

			// 2) get the subgradient
			// psi_i(y) := phi(x_i,y_i) - phi(x_i, y_pred)
			SGVector<float64_t> psi_i(M);
			if (result->psi_computed)
			{
				SGVector<float64_t>::add(psi_i.vector,
					1.0, result->psi_truth.vector, -1.0, result->psi_pred.vector,
					psi_i.vlen);
			}
			else if(result->psi_computed_sparse)
			{
				psi_i.zero();
				result->psi_pred_sparse.add_to_dense(1.0, psi_i.vector, psi_i.vlen);
				result->psi_truth_sparse.add_to_dense(-1.0, psi_i.vector, psi_i.vlen);
			}
			else
			{
				SG_ERROR("model(%s) should have either of psi_computed or psi_computed_sparse"
						"to be set true\n", m_model->get_name());
			}

			// 3) loss_i = L(y_i, y_pred)
			float64_t loss_i = result->delta;
			ASSERT(loss_i - CMath::dot(m_w.vector, psi_i.vector, m_w.vlen) >= -1e-12);

			// 4) update w_s and ell_s
			add<linalg::Backend::NATIVE>(w_s, psi_i, w_s);
			ell_s += loss_i;

			SG_UNREF(result);

		} // end si

		scale<linalg::Backend::NATIVE>(w_s, 1.0 / (N*m_lambda));

		ell_s /= N;

		// 5) duality gap
		SGVector<float64_t> w_diff = m_w.clone();
		SGVector<float64_t>::add(w_diff.vector, 1.0, m_w.vector, -1.0, w_s.vector, w_s.vlen);
		float64_t dual_gap = m_lambda * CMath::dot(m_w.vector, w_diff.vector, m_w.vlen) - m_ell + ell_s;

		// Debug: compute primal and dual objectives and training error
		if (m_verbose)
		{
			float64_t primal = CSOSVMHelper::primal_objective(m_w, m_model, m_lambda);
			float64_t dual = CSOSVMHelper::dual_objective(m_w, m_ell, m_lambda);
			ASSERT(CMath::fequals_abs(primal - dual, dual_gap, 1e-12));
			float64_t train_error = CSOSVMHelper::average_loss(m_w, m_model); // Note train_error isn't ell_s

			SG_SPRINT("pass %d (iteration %d), primal = %f, dual = %f, duality gap = %f, train_error = %f \n",
				pi, k, primal, dual, dual_gap, train_error);

			m_helper->add_debug_info(primal, (1.0*k) / N, train_error, dual, dual_gap);
		}

		// 6) check duality gap
		if (dual_gap <= m_gap_threshold)
		{
			SG_DEBUG("iteration %d...\n", k);
			SG_DEBUG("current gap: %f, gap_threshold: %f\n", dual_gap, m_gap_threshold);
			SG_DEBUG("Duality gap below threshold -- stopping!\n");
			break; // stop main loop
		}
		else
		{
			SG_DEBUG("iteration %d...\n", k);
			SG_DEBUG("current gap: %f.\n", dual_gap);
		}

		// 7) step-size gamma
		float64_t gamma = 1.0 / (k+1.0);
		if (m_do_line_search)
		{
			gamma = dual_gap / (m_lambda \
					* (CMath::dot(w_diff.vector, w_diff.vector, w_diff.vlen) + 1e-12));
			gamma = ((gamma > 1 ? 1 : gamma) < 0) ? 0 : gamma; // clip to [0,1], or max(0,min(1,gamma))
		}

		// 8) finally update w and ell
		SGVector<float64_t>::add(m_w.vector, 1.0-gamma, m_w.vector, gamma, w_s.vector, m_w.vlen);
		m_ell = (1.0-gamma) * m_ell + gamma * ell_s;

	} // end pi

	if (m_verbose)
		m_helper->terminate();

	SG_DEBUG("Leaving CFWSOSVM::train_machine.\n");
	return true;
}

float64_t CFWSOSVM::get_lambda() const
{
	return m_lambda;
}

void CFWSOSVM::set_lambda(float64_t lbda)
{
	m_lambda = lbda;
}

int32_t CFWSOSVM::get_num_iter() const
{
	return m_num_iter;
}

void CFWSOSVM::set_num_iter(int32_t num_iter)
{
	m_num_iter = num_iter;
}

float64_t CFWSOSVM::get_gap_threshold() const
{
	return m_gap_threshold;
}

void CFWSOSVM::set_gap_threshold(float64_t gap_threshold)
{
	m_gap_threshold = gap_threshold;
}

float64_t CFWSOSVM::get_ell() const
{
	return m_ell;
}

void CFWSOSVM::set_ell(float64_t ell)
{
	m_ell = ell;
}

