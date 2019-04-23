/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Abinash Panda, Shell Hu, Soeren Sonnenburg, Fernando Iglesias,
 *          Bjoern Esser
 */

#include <shogun/base/progress.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/structure/StochasticSOSVM.h>
#include <shogun/mathematics/UniformIntDistribution.h>

using namespace shogun;

StochasticSOSVM::StochasticSOSVM()
: RandomMixin<LinearStructuredOutputMachine>()
{
	init();
}

StochasticSOSVM::StochasticSOSVM(
		std::shared_ptr<StructuredModel>  model,
		std::shared_ptr<StructuredLabels> labs,
		bool do_weighted_averaging,
		bool verbose)
: RandomMixin<LinearStructuredOutputMachine>(model, labs)
{
	require(model != NULL && labs != NULL,
		"{}::CStochasticSOSVM(): model and labels cannot be NULL!", get_name());

	require(labs->get_num_labels() > 0,
		"{}::CStochasticSOSVM(): number of labels should be greater than 0!", get_name());

	init();
	m_lambda = 1.0 / labs->get_num_labels();
	m_do_weighted_averaging = do_weighted_averaging;
	m_verbose = verbose;
}

void StochasticSOSVM::init()
{
	SG_ADD(&m_lambda, "lambda", "Regularization constant");
	SG_ADD(&m_num_iter, "num_iter", "Number of iterations");
	SG_ADD(&m_do_weighted_averaging, "do_weighted_averaging", "Do weighted averaging");
	SG_ADD(&m_debug_multiplier, "debug_multiplier", "Debug multiplier");

	m_lambda = 1.0;
	m_num_iter = 50;
	m_do_weighted_averaging = true;
	m_debug_multiplier = 0;
}

StochasticSOSVM::~StochasticSOSVM()
{
}

EMachineType StochasticSOSVM::get_classifier_type()
{
	return CT_STOCHASTICSOSVM;
}

bool StochasticSOSVM::train_machine(std::shared_ptr<Features> data)
{
	SG_TRACE("Entering CStochasticSOSVM::train_machine.");
	if (data)
		set_features(data);

	// Initialize the model for training
	m_model->init_training();
	// Check that the scenary is correct to start with training
	m_model->check_training_setup();
	SG_DEBUG("The training setup is correct.");

	// Dimensionality of the joint feature space
	int32_t M = m_model->get_dim();
	// Number of training examples
	int32_t N = m_labels->as<StructuredLabels>()->get_num_labels();

	SG_DEBUG("M={}, N ={}.", M, N);

	// Initialize the weight vector
	m_w = SGVector<float64_t>(M);
	m_w.zero();

	SGVector<float64_t> w_avg;
	if (m_do_weighted_averaging)
		w_avg = m_w.clone();

	// logging
	if (m_verbose)
	{
		if (m_helper != NULL)


		m_helper = std::make_shared<SOSVMHelper>();

	}

	int32_t debug_iter = 1;
	if (m_debug_multiplier == 0)
	{
		debug_iter = N;
		m_debug_multiplier = 100;
	}

	// Main loop
	int32_t k = 0;
	UniformIntDistribution<int32_t> uniform_int_dist;
	for (auto pi : SG_PROGRESS(range(m_num_iter)))
	{
		for (int32_t si = 0; si < N; ++si)
		{
			// 1) Picking random example
			int32_t i = uniform_int_dist(m_prng, {0, N-1});

			// 2) solve the loss-augmented inference for point i
			auto result = m_model->argmax(m_w, i);

			// 3) get the subgradient
			// psi_i(y) := phi(x_i,y_i) - phi(x_i, y)
			SGVector<float64_t> psi_i(M);
			SGVector<float64_t> w_s(M);

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
				error("model({}) should have either of psi_computed or psi_computed_sparse"
						"to be set true", m_model->get_name());
			}

			w_s = psi_i.clone();
			w_s.scale(1.0 / (N*m_lambda));

			// 4) step-size gamma
			float64_t gamma = 1.0 / (k+1.0);

			// 5) finally update the weights
			SGVector<float64_t>::add(m_w.vector,
				1.0-gamma, m_w.vector, gamma*N, w_s.vector, m_w.vlen);

			// 6) Optionally, update the weighted average
			if (m_do_weighted_averaging)
			{
				float64_t rho = 2.0 / (k+2.0);
				SGVector<float64_t>::add(w_avg.vector,
					1.0-rho, w_avg.vector, rho, m_w.vector, w_avg.vlen);
			}

			k += 1;


			// Debug: compute objective and training error
			if (m_verbose && k == debug_iter)
			{
				SGVector<float64_t> w_debug;
				if (m_do_weighted_averaging)
					w_debug = w_avg.clone();
				else
					w_debug = m_w.clone();

				float64_t primal = SOSVMHelper::primal_objective(w_debug, m_model, m_lambda);
				float64_t train_error = SOSVMHelper::average_loss(w_debug, m_model);

				SG_DEBUG("pass {} (iteration {}), SVM primal = {}, train_error = {} ",
					pi, k, primal, train_error);

				m_helper->add_debug_info(primal, (1.0*k) / N, train_error);

				debug_iter = Math::min(debug_iter+N, debug_iter*(1+m_debug_multiplier/100));
			}
		}
	}

	if (m_do_weighted_averaging)
		m_w = w_avg.clone();

	if (m_verbose)
		m_helper->terminate();

	SG_TRACE("Leaving CStochasticSOSVM::train_machine.");
	return true;
}

float64_t StochasticSOSVM::get_lambda() const
{
	return m_lambda;
}

void StochasticSOSVM::set_lambda(float64_t lbda)
{
	m_lambda = lbda;
}

int32_t StochasticSOSVM::get_num_iter() const
{
	return m_num_iter;
}

void StochasticSOSVM::set_num_iter(int32_t num_iter)
{
	m_num_iter = num_iter;
}

int32_t StochasticSOSVM::get_debug_multiplier() const
{
	return m_debug_multiplier;
}

void StochasticSOSVM::set_debug_multiplier(int32_t multiplier)
{
	m_debug_multiplier = multiplier;
}

