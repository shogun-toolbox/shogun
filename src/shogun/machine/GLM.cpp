/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Tej Sukhatme
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/machine/GLM.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/RandomNamespace.h>
// #include <utility>
#include <cmath>

using namespace shogun;

GLM::GLM(GLM_DISTRIBUTION distr, float64_t alpha, float64_t lambda, float64_t learning_rate, int32_t max_iterations, float64_t tolerance, float64_t eta): IterativeMachine<LinearMachine>()
{
	init();
	distribution=distr;
	m_alpha=alpha;
	m_lambda=lambda;
	m_learning_rate=learning_rate;
	m_max_iterations=max_iterations;
	m_tolerance=tolerance;
	m_eta=eta;
}

void GLM::init()
{
	bias = 0;

	distribution= POISSON;
	m_alpha=0.5;
	m_lambda=0.1;
	m_learning_rate=2e-1;
	m_max_iterations=1000;
	m_tolerance=1e-6;
	m_eta=2.0;
}


GLM::~GLM()
{

}

std::shared_ptr<RegressionLabels> GLM::apply_regression(std::shared_ptr<Features> data)
{
	LinearMachine::set_features(std::static_pointer_cast<DotFeatures>(data));

	if (!LinearMachine::features)
		return std::make_shared<RegressionLabels>(SGVector<float64_t>());

	int32_t num=LinearMachine::features->get_num_vectors();
	ASSERT(num>0)
	ASSERT(m_w.vlen==features->get_dim_feature_space())
	SGVector<float64_t> out(num);
	LinearMachine::features->dense_dot_range(out.vector, 0, num, NULL, m_w.vector, m_w.vlen, bias);
	SGVector<float64_t> result = non_linearity(out);
	return std::make_shared<RegressionLabels>(result);
}

void GLM::init_model(std::shared_ptr<Features> data)
{
	LinearMachine::set_features(std::static_pointer_cast<DotFeatures>(data));
	SGMatrix<float64_t> X;
	SGVector<float64_t> y;
	NormalDistribution<float64_t> normal_dist;
	auto n_features = LinearMachine::features->get_dim_feature_space();
	
	if (m_compute_bias && bias == 0)
		bias = 1 / (n_features + 1) * normal_dist(m_prng);

	if(!m_w.data())
	{
		m_w = SGVector<float64_t>(n_features);
		for (int i = 0; i < n_features; i++)
			m_w = 1 / (n_features + 1) * normal_dist(m_prng);
	}
}

void GLM::iteration()
{
	SGMatrix<float64_t> X = LinearMachine::features->get_computed_dot_feature_matrix();
	SGVector<float64_t> y = m_labels->get_values();
	SGVector<float64_t> w_old(m_w);
	SGVector<float64_t> gradient_w = compute_grad_L2_loss_w(X, y, m_w, bias);
	float64_t gradient_bias = compute_grad_L2_loss_bias(X, y, m_w, bias);

	//Update
	m_w = linalg::add(m_w, gradient_w, 1.0, -1*m_learning_rate);
	if(m_compute_bias)
		bias = bias - m_learning_rate * gradient_bias;
	
	//Apply proximal operator
	m_w = apply_proximal_operator(m_w, m_lambda * m_alpha);

	//Convergence by relative parameter change tolerance
	float64_t norm_update = linalg::norm(linalg::add(m_w, w_old, 1.0, -1.0));
	if(m_current_iteration > 0 && (norm_update / linalg::norm(m_w)) < m_tolerance)
		m_complete = true;
}

void GLM::set_tau(SGMatrix<float64_t> tau)
{
	m_tau = tau;
}

SGMatrix<float64_t> GLM::get_tau()
{
	return m_tau;
}

const SGVector<float64_t> GLM::conditional_intensity(const SGMatrix<float64_t> X, const SGVector<float64_t> w,const float64_t l_bias)
{
	SGVector<float64_t> z = compute_z(X, w, l_bias);
	SGVector<float64_t> result = non_linearity(z);
	return result;
}

const SGVector<float64_t> GLM::compute_z(const SGMatrix<float64_t> X, const SGVector<float64_t> w, const float64_t l_bias)
{
	SGVector<float64_t> z = linalg::matrix_prod(X, w, false);
	return z;
}

const SGVector<float64_t> GLM::non_linearity(const SGVector<float64_t> z)
{
	SGVector<float64_t> result;
	switch (distribution)
	{
	case POISSON:
		result = SGVector<float64_t>(z);
		float64_t l_bias = 0;

		if(LinearMachine::m_compute_bias)
			l_bias = (1 - m_eta) * std::exp(m_eta);

		for (int i = 0; i < z.vlen; i++)
		{
			if(z[i]>m_eta)
				result[i] = z[i] * std::exp(m_eta) + l_bias;
			else
				result[i] = std::exp(z[i]);
		}
		break;
	
	default:
		error("Not a valid distribution type");
		break;
	}
	return result;
}

const SGVector<float64_t> GLM::gradient_non_linearity(const SGVector<float64_t> z)
{
	SGVector<float64_t> result;
	switch (distribution)
	{
	case POISSON:
		result = SGVector<float64_t>(z);
		for (int i = 0; i < z.vlen; i++)
		{
			if(z[i]>m_eta)
				
				result[i] = std::exp(m_eta);
			else
				result[i] = std::exp(z[i]);
		}
		break;
	
	default:
		error("Not a valid distribution type");
		break;
	}
	return result;
}

const SGVector<float64_t> GLM::compute_grad_L2_loss_w(const SGMatrix<float64_t> X, const SGVector<float64_t> y, const SGVector<float64_t> w, const float64_t l_bias)
{
	auto n_samples = y.vlen;
	auto n_features = X.num_rows;
	if(!m_tau.data())
		m_tau = SGMatrix<float64_t>::create_identity_matrix(w.vlen, 1.0);
	SGMatrix<float64_t> inv_cov = linalg::matrix_prod(m_tau, m_tau, true, false);
	
	SGVector<float64_t> z = compute_z(X, w, l_bias);
	SGVector<float64_t> mu = non_linearity(z);
	SGVector<float64_t> grad_mu = gradient_non_linearity(z);

	SGVector<float64_t> grad_w(w.vlen);
	switch (distribution)
	{
	case POISSON:
		SGVector<float64_t> a = y * grad_mu / mu;
		linalg::transpose_matrix(linalg::add(linalg::matrix_prod(SGMatrix(grad_mu), X, true, false)), (linalg::matrix_prod(SGMatrix(a), X, true, false)), 1, -1);
		break;
	
	default:
		error("Not a valid distribution type");
		break;
	}
	for (int i = 0; i < grad_w.vlen; i++)
		grad_w[i] /= n_samples;
	
	linalg::add(grad_w, linalg::matrix_prod(inv_cov, w, false), 1.0, m_lambda * (1 - m_alpha));

	return grad_w;
}

const float64_t GLM::compute_grad_L2_loss_bias(const SGMatrix<float64_t> X, const SGVector<float64_t> y, const SGVector<float64_t> w, const float64_t l_bias)
{
	auto n_samples = y.vlen;
	SGVector<float64_t> z = compute_z(X, w, l_bias);
	SGVector<float64_t> mu = non_linearity(z);
	SGVector<float64_t> grad_mu = gradient_non_linearity(z);

	float64_t grad_bias = 0;
	switch (distribution)
	{
	case POISSON:
		for (int i = 0; i < grad_mu.vlen; i++)
		{
			grad_bias += grad_mu[i];
			grad_bias -= y[i] * grad_mu[i] / mu[i];
		}
		break;
	
	default:
		error("Not a valid distribution type");
		break;
	}
	grad_bias /= n_samples;

	return grad_bias;
}

const SGVector<float64_t> GLM::apply_proximal_operator(const SGVector<float64_t> w, const float64_t threshold)
{
	SGVector<float64_t> result(w);
	for (int i = 0; i < w.vlen; i++)
	{
		result[i] = 0;
		if(w[i]>0 && abs(w[i])>threshold)
			result[i] = abs(w[i]) - threshold;
		else if(w[i]<0 && abs(w[i])>threshold)
			result[i] = -1* (abs(w[i]) - threshold);
	}
	return result;
}
