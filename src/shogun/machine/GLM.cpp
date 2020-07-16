/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Author: Tej Sukhatme
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/machine/GLM.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/RandomNamespace.h>
#include <shogun/optimization/ElasticNetPenalty.h>
#include <shogun/optimization/SGDMinimizer.h>
#include <shogun/optimization/GradientDescendUpdater.h>
#include <shogun/optimization/ConstLearningRate.h>

#include<iostream>
// #include <utility>
#include <cmath>

using namespace shogun;





GLM::GLM()
{
	// std::cout<<"Entered non-parameterized constructor.\n";
	SG_ADD(&distribution, "distribution_type", "variable to store name of distribution type", ParameterProperties::HYPER);
	SG_ADD(&m_eta, "eta", "threshold parameter that linearizes the exp() function above eta", ParameterProperties::HYPER);
	SG_ADD(&m_lambda, "lambda", "regularization parameter of penalty term", ParameterProperties::HYPER);
	SG_ADD(&m_alpha, "alpha", "weighting between L1 penalty and L2 penalty term of the loss function", ParameterProperties::HYPER);
	SG_ADD(&m_tolerance, "tolerance", "convergence threshold or stopping criteria", ParameterProperties::HYPER);
	
	m_gradient_updater = std::make_shared<GradientDescendUpdater>();
	m_learning_rate = std::make_shared<ConstLearningRate>();
	m_penalty = std::make_shared<ElasticNetPenalty>();
	m_cost_function = std::make_shared<GLMCostFunction>();
	// std::cout<<"Exiting non-parameterized constructor.\n";
}

GLM::GLM(GLM_DISTRIBUTION distr, float64_t alpha, float64_t lambda, float64_t learning_rate, int32_t max_iterations, float64_t tolerance, float64_t eta): GLM()
{
	// std::cout<<"Entered parameterized constructor.\n";
	distribution=distr;
	m_alpha=alpha;
	m_lambda=lambda;
	m_max_iterations=max_iterations;
	m_tolerance=tolerance;
	m_eta=eta;

	m_learning_rate->set_const_learning_rate(learning_rate);

	m_penalty->set_l1_ratio(m_alpha);
	// std::cout<<"Exiting parameterized constructor.\n";
}

std::shared_ptr<RegressionLabels> GLM::apply_regression(std::shared_ptr<Features> data)
{
	// std::cout<<"Entered apply_regression().\n";

	LinearMachine::set_features(std::static_pointer_cast<DotFeatures>(data));

	if (!LinearMachine::features)
		return std::make_shared<RegressionLabels>(SGVector<float64_t>());

	auto num = LinearMachine::features->get_num_vectors();
	ASSERT(num>0)
	ASSERT(m_w.vlen==features->get_dim_feature_space())
	SGVector<float64_t> out(num);
	LinearMachine::features->dense_dot_range(out.vector, 0, num, NULL, m_w.vector, m_w.vlen, bias);
	auto result = m_cost_function->non_linearity(out);
	
	// std::cout<<"Exiting apply_regression().\n";
	return std::make_shared<RegressionLabels>(result);
}

void GLM::init_model(const std::shared_ptr<Features>& data)
{
	// std::cout<<"Entered init_model().\n";
	ASSERT(m_labels)
	if (data)
	{
		if (!data->has_property(FP_DOT))
			error("Specified features are not of type CDotFeatures");
		LinearMachine::set_features(std::static_pointer_cast<DotFeatures>(data));
	}
	ASSERT(features)

	NormalDistribution<float64_t> normal_dist;
	auto n_features = LinearMachine::features->get_dim_feature_space();

	// std::cout<<"Number of features: "<<n_features<<'\n';
	// std::cout<<"No of elements in m_w"<<m_w.vlen<<'\n';
	if(m_w.vlen == 0)
	{
		if (m_compute_bias && bias == 0)
			bias = 1.0 / (n_features + 1) * normal_dist(m_prng);

		// std::cout<<"Bias: "<<bias<<'\n';
		if(n_features > 0)
		{
			m_w = SGVector<float64_t>(n_features);
			for (int i = 0; i < n_features; i++)
			{
				// std::cout<<"\tFor feature number: "<<i<<'\t';
				auto rand = normal_dist(m_prng);
				// std::cout<<"\tRandom Value is "<<rand<<'\t';
				m_w[i] = 1.0 / (n_features + 1) * rand;
				// std::cout<<"\tSetting value "<<m_w[i]<<'\n';
			}
		}
	}
	m_cost_function->set_target(shared_from_this()->as<GLM>());

	// std::cout<<"Exiting init_model().\n";
}

void GLM::iteration()
{
	// std::cout<<"Entered iteration().\n";
	//std::shared_ptr<GLMCostFunction> m_cost_function;
	//std::cout<<"Iteration Number: "<<m_current_iteration<<'\n';
	//std::cout<<"Completed: "<<m_complete<<'\n';
	auto learning_rate = m_learning_rate->get_learning_rate(m_current_iteration);
	SGVector<float64_t> w_old(m_w.vlen);
	for(int i = 0; i<m_w.vlen; i++)
		w_old[i] = m_w[i];
	m_w.display_vector("Weights");
	std::cout<<"Bias for iteration "<<m_current_iteration<<" is "<<bias<<'\n';
	auto gradient_w = m_cost_function->get_gradient();
	gradient_w.display_vector("weights gradient");
	auto gradient_bias = m_cost_function->get_gradient_bias();
	std::cout<<"gradient_bias for iteration "<<m_current_iteration<<" is "<<gradient_bias<<'\n';
	// std::cout<<"Calculating new weights.\n";
	//for (int i = 0; i<m_w.vlen; i++)
	//	m_w[i] += m_lambda * m_penalty->get_penalty_gradient(m_w[i], 0.0);
	// std::cout<<"Penalty added.\n";
	//Update
	// m_gradient_updater->update_variable(m_w, gradient_w, learning_rate);

	std::cout<<"Learning rate is "<<learning_rate<<'\n';
	m_w = linalg::add(m_w, gradient_w, 1.0, -1 * learning_rate);
	// std::cout<<"Calculated new weights.\n";

	if(m_compute_bias)
		bias -= learning_rate * gradient_bias;

	//Apply proximal operator
	// m_penalty->update_variable_for_proximity(m_w, m_lambda * m_alpha);
	// beta = np.sign(beta) * (np.abs(beta) - thresh) * \
    //             (np.abs(beta) > thresh)
	m_w.display_vector("Before proximal operator");
	for(int i = 0; i < m_w.vlen; i++)
	{
		if(abs(m_w[i]) < (m_lambda * m_alpha))
			m_w[i] = 0;
		else {
			if(m_w[i]>0)
				m_w[i] = m_w[i] - (m_lambda * m_alpha);
			else
				m_w[i] = m_w[i] + (m_lambda * m_alpha);
		}
	}
	m_w.display_vector("After proximal operator");
	//Convergence by relative parameter change tolerance
	// w_old.display_vector("Old weights");
	// m_w.display_vector("New weights");
	auto norm_update = linalg::norm(linalg::add(m_w, w_old, 1.0, -1.0));
	float32_t checker = linalg::norm(m_w)==0 ? norm_update : abs(norm_update / linalg::norm(m_w));
	if(m_current_iteration > 0 && checker < m_tolerance){
		// std::cout<<"Entered if statement.\n";
		m_complete = true;
	}
		

	// std::cout<<"Exiting iteration().\n";
}
