/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Author: Tej Sukhatme
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/machine/GLM.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/RandomNamespace.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/optimization/ConstLearningRate.h>
#include <shogun/optimization/ElasticNetPenalty.h>
#include <shogun/optimization/GradientDescendUpdater.h>
#include <shogun/optimization/SGDMinimizer.h>

#include <cmath>

using namespace shogun;

GLM::GLM()
{
	SG_ADD_OPTIONS(
	    (machine_int_t*)&distribution, "distribution_type",
	    "variable to store name of distribution type",
	    ParameterProperties::HYPER, SG_OPTIONS(POISSON));
	SG_ADD(
	    &m_eta, "eta",
	    "threshold parameter that linearizes the exp() function above eta",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_lambda, "lambda", "regularization parameter of penalty term",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_alpha, "alpha",
	    "weighting between L1 penalty and L2 penalty term of the loss function",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_tolerance, "tolerance", "convergence threshold or stopping criteria",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_learning_rate, "learning_rate", "learning rate for gradient descent",
	    ParameterProperties::HYPER);

	m_gradient_updater = std::make_shared<GradientDescendUpdater>();
	m_penalty = std::make_shared<ElasticNetPenalty>();
	m_cost_function = std::make_shared<GLMCostFunction>();
}

GLM::GLM(
    GLM_DISTRIBUTION distr, float64_t alpha, float64_t lambda,
    float64_t learning_rate, int32_t max_iterations, float64_t tolerance,
    float64_t eta)
    : GLM()
{
	distribution = distr;
	m_alpha = alpha;
	m_lambda = lambda;
	m_learning_rate = learning_rate;
	m_max_iterations = max_iterations;
	m_tolerance = tolerance;
	m_eta = eta;

	m_penalty->set_l1_ratio(m_alpha);
}

std::shared_ptr<RegressionLabels>
GLM::apply_regression(std::shared_ptr<Features> data)
{
	std::shared_ptr<DotFeatures> features;
	if (data)
	{
		if (!data->has_property(FP_DOT))
			error("Specified features are not of type CDotFeatures");
		features = std::static_pointer_cast<DotFeatures>(data);
	}

	auto num = features->get_num_vectors();
	ASSERT(num > 0)
	ASSERT(m_w.vlen == features->get_dim_feature_space())
	SGVector<float64_t> out(num);
	features->dense_dot_range(
	    out.vector, 0, num, NULL, m_w.vector, m_w.vlen, bias);
	auto result = m_cost_function->non_linearity(
	    out, m_compute_bias, m_eta, distribution);

	return std::make_shared<RegressionLabels>(result);
}

void GLM::init_model(const std::shared_ptr<DotFeatures>& data)
{
	NormalDistribution<float64_t> normal_dist;
	const auto& n_features = data->get_dim_feature_space();

	if (m_w.vlen == 0)
	{
		if (m_compute_bias && bias == 0)
			bias = 1.0 / (n_features + 1) * normal_dist(m_prng);

		if (n_features > 0)
		{
			m_w = SGVector<float64_t>(n_features);

			std::generate(m_w.begin(), m_w.end(), [&]() {
				auto rand = normal_dist(m_prng);
				return 1.0 / (n_features + 1) * rand;
			});
		}
	}
}

void GLM::iteration(const std::shared_ptr<DotFeatures>& features, 
	const std::shared_ptr<Labels>& labs)
{
	SGVector<float64_t> w_old = m_w.clone();

	auto X = features->get_computed_dot_feature_matrix();
	auto y = regression_labels(labs)->get_labels();

	auto gradient_w = m_cost_function->get_gradient_weights(
	    X, y, m_w, bias, m_lambda, m_alpha, m_compute_bias, m_eta,
	    distribution);
	auto gradient_bias = m_cost_function->get_gradient_bias(
	    X, y, m_w, bias, m_compute_bias, m_eta, distribution);

	// Update
	// TODO: Use gradient updater
	// m_gradient_updater->update_variable(m_w, gradient_w, learning_rate);
	m_w = linalg::add(m_w, gradient_w, 1.0, -1 * m_learning_rate);

	if (m_compute_bias)
		bias -= m_learning_rate * gradient_bias;

	// Apply proximal operator
	// TODO: Use proximity updater.
	// m_penalty->update_variable_for_proximity(m_w, m_lambda * m_alpha);
	for (auto i : range(m_w.vlen))
	{
		if (std::abs(m_w[i]) < (m_lambda * m_alpha))
			m_w[i] = 0;
		else
		{
			if (m_w[i] > 0)
				m_w[i] -= (m_lambda * m_alpha);
			else
				m_w[i] += (m_lambda * m_alpha);
		}
	}

	// Convergence by relative parameter change tolerance
	auto norm_update = linalg::norm(linalg::add(m_w, w_old, 1.0, -1.0));
	float32_t checker = linalg::norm(m_w) == 0
	                        ? norm_update
	                        : std::abs(norm_update / linalg::norm(m_w));
	if (m_current_iteration > 0 && checker < m_tolerance)
		m_complete = true;
}
