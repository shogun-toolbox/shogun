/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Tej Sukhatme
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
// #include <utility>
#include <cmath>

using namespace shogun;

GLM::GLM()
{
	SG_ADD(&distribution, "distribution_type", "variable to store name of distribution type", ParameterProperties::HYPER);
	SG_ADD(&m_eta, "eta", "threshold parameter that linearizes the exp() function above eta", ParameterProperties::HYPER);
	SG_ADD(&m_lambda, "lambda", "regularization parameter of penalty term", ParameterProperties::HYPER);
	SG_ADD(&m_alpha, "alpha", "weighting between L1 penalty and L2 penalty term of the loss function", ParameterProperties::HYPER);
	SG_ADD(&m_tolerance, "tolerance", "convergence threshold or stopping criteria", ParameterProperties::HYPER);
	
}

GLM::GLM(GLM_DISTRIBUTION distr, float64_t alpha, float64_t lambda, float64_t learning_rate, int32_t max_iterations, float64_t tolerance, float64_t eta): IterativeMachine<LinearMachine>()
: GLM()
{
	distribution=distr;
	m_alpha=alpha;
	m_lambda=lambda;
	m_max_iterations=max_iterations;
	m_tolerance=tolerance;
	m_eta=eta;

	m_learning_rate->set_const_learning_rate(learning_rate);
	
	m_cost_function->set_target(std::make_shared<GLM> (this));

	m_penalty->set_l1_ratio(m_alpha);
}

std::shared_ptr<RegressionLabels> GLM::apply_regression(std::shared_ptr<Features> data)
{
	LinearMachine::set_features(std::static_pointer_cast<DotFeatures>(data));

	if (!LinearMachine::features)
		return std::make_shared<RegressionLabels>(SGVector<float64_t>());

	auto num = LinearMachine::features->get_num_vectors();
	ASSERT(num>0)
	ASSERT(m_w.vlen==features->get_dim_feature_space())
	SGVector<float64_t> out(num);
	LinearMachine::features->dense_dot_range(out.vector, 0, num, NULL, m_w.vector, m_w.vlen, bias);
	auto result = m_cost_function->non_linearity(out);
	return std::make_shared<RegressionLabels>(result);
}

void GLM::init_model(const std::shared_ptr<Features>& data)
{
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
	
	if (m_compute_bias && bias == 0)
		bias = 1 / (n_features + 1) * normal_dist(m_prng);

	if(m_w.size() > 0)
	{
		m_w = SGVector<float64_t>(n_features);
		for (auto& weight : m_w)
			weight = 1 / (n_features + 1) * normal_dist(m_prng);
	}
}

void GLM::iteration()
{
	auto learning_rate = m_learning_rate->get_learning_rate(m_current_iteration);
	SGVector<float64_t> w_old(m_w);
	auto gradient_w = m_cost_function->get_gradient();
	auto gradient_bias = m_cost_function->get_gradient_bias();

	for (auto& weight : m_w)
		weight += m_lambda * m_penalty->get_penalty_gradient(weight, 0.0);

	//Update
	m_gradient_updater->update_variable(m_w, gradient_w, learning_rate);
	m_w = linalg::add(m_w, gradient_w, 1.0, -1 * learning_rate);
	if(m_compute_bias)
		bias -= learning_rate * gradient_bias;
	
	//Apply proximal operator
	m_penalty->update_variable_for_proximity(m_w, m_lambda * m_alpha);

	//Convergence by relative parameter change tolerance
	auto norm_update = linalg::norm(linalg::add(m_w, w_old, 1.0, -1.0));
	if(m_current_iteration > 0 && (norm_update / linalg::norm(m_w)) < m_tolerance)
		m_complete = true;
}



class GLMCostFunction: public FirstOrderCostFunction
{
public:

	friend class GLM;

	void set_target(const std::shared_ptr<GLM>&obj)
	{
		if(m_obj != obj)
			m_obj=obj;
	}

	void unset_target()
	{
		m_obj=NULL;
	}

	virtual float64_t get_cost()
	{
		//TODO
	}

	virtual SGVector<float64_t> obtain_variable_reference()
	{
		require(m_obj,"Object not set");
		return m_obj->m_w;
	}

	virtual SGVector<float64_t> get_gradient()
	{
		auto X = m_obj->LinearMachine::features->get_computed_dot_feature_matrix();
		auto y = m_obj->m_labels->get_values();
		SGVector<float64_t> w_old(m_obj->m_w);

		auto n_samples = y.vlen;
		auto n_features = X.num_rows;
	
		auto z = compute_z(X, m_obj->m_w, m_obj->bias);
		auto mu = non_linearity(z);
		auto grad_mu = gradient_non_linearity(z);

		SGVector<float64_t> grad_w(m_obj->m_w.vlen);
		SGVector<float64_t> a;
		switch (m_obj->distribution)
		{
		case POISSON:
			a = linalg::element_prod(y, grad_mu);
			for(int i = 0; i<y.vlen; i++)
				a[i] /= mu[i];
			linalg::transpose_matrix(linalg::add(linalg::matrix_prod(SGMatrix(grad_mu), X, true, false), (linalg::matrix_prod(SGMatrix(a), X, true, false)), 1.0, -1.0));
			break;
	
		default:
			error("Distribution type {} not implemented.", m_obj->distribution);
			break;
		}

		grad_w = linalg::scale(grad_w, 1.0/n_samples);	
		return grad_w;
	}

	virtual float64_t get_gradient_bias()
	{
		auto X = m_obj->LinearMachine::features->get_computed_dot_feature_matrix();
		auto y = m_obj->m_labels->get_values();

		auto n_samples = y.vlen;
		auto z = compute_z(X, m_obj->m_w, m_obj->bias);
		auto mu = non_linearity(z);
		auto grad_mu = gradient_non_linearity(z);

		float64_t grad_bias = 0;
		switch (m_obj->distribution)
		{
		case POISSON:
			for (int i = 0; i < grad_mu.vlen; i++)
			{
				grad_bias += grad_mu[i];
				grad_bias -= y[i] * grad_mu[i] / mu[i];
			}
			break;
	
		default:
			error("Distribution type {} not implemented.", m_obj->distribution);
			break;
		}
		grad_bias /= n_samples;

		return grad_bias;
	}

	virtual const char* get_name() const { return "GLMCostFunction"; }

private:

	virtual const SGVector<float64_t> compute_z(const SGMatrix<float64_t> X, const SGVector<float64_t> w, const float64_t bias)
	{
		return linalg::matrix_prod(X, w, false);
	}

	virtual const SGVector<float64_t> non_linearity(const SGVector<float64_t> z)
	{
		SGVector<float64_t> result;
		switch (m_obj->distribution)
		{
		case POISSON:
			result = SGVector<float64_t>(z);
			float64_t l_bias = 0;

			if(m_obj->LinearMachine::m_compute_bias)
				l_bias = (1 - m_obj->m_eta) * std::exp(m_obj->m_eta);

			for (int i = 0; i < z.vlen; i++)
			{
				if(z[i]>m_obj->m_eta)
					result[i] = z[i] * std::exp(m_obj->m_eta) + l_bias;
				else
					result[i] = std::exp(z[i]);
			}
			break;
	
		default:
			error("Distribution type {} not implemented.", m_obj->distribution);
			break;
		}
		return result;
	}

	virtual const SGVector<float64_t> gradient_non_linearity(const SGVector<float64_t> z)
	{
		SGVector<float64_t> result;
		switch (m_obj->distribution)
		{
		case POISSON:
			result = SGVector<float64_t>(z);
			for (int i = 0; i < z.vlen; i++)
			{
				if(z[i]>m_obj->m_eta)
					result[i] = std::exp(m_obj->m_eta);
				else
					result[i] = std::exp(z[i]);
			}
			break;

		default:
			error("Distribution type {} not implemented.", m_obj->distribution);
			break;
		}
		return result;
	}

	std::shared_ptr<GLM>m_obj;
};
