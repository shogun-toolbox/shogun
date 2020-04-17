/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Tej Sukhatme
 */

// #include <rxcpp/rx-lite.hpp>
// #include <shogun/features/DotFeatures.h>
// #include <shogun/labels/Labels.h>
// #include <shogun/labels/RegressionLabels.h>
#include <shogun/machine/GeneralizedLinearMachine.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/RandomNamespace.h>
// #include <utility>
#include <cmath>

using namespace shogun;

GeneralizedLinearMachine::GeneralizedLinearMachine(): LinearMachine()
{
	init();
}

void GeneralizedLinearMachine::init()
{
	bias = 0;
	features = NULL;
	m_w = NULL;

	distribution= POISSON;
	m_alpha=0.5;
	m_tau=NULL;
	m_lambda=0.1;
	m_learning_rate=2e-1;
	m_max_iter=1000;
	m_tolerance=1e-6;
	m_eta=2.0;
	m_fit_intercept=true;
}


GeneralizedLinearMachine::~GeneralizedLinearMachine()
{

}

SGVector<float64_t> GeneralizedLinearMachine::predict(SGMatrix<float64_t> X)
{
	SGVector<float64_t> result = conditional_intensity(X, m_w, bias);
	return result;
}

bool GeneralizedLinearMachine::fit(SGMatrix<float64_t> X, SGVector<float64_t> y)
{
	//Initialize parameters
	NormalDistribution<float64_t> normal_dist;
	auto n_features = X.num_rows;
	SGVector<float64_t> w;
	int l_bias = bias;
	
	if (m_fit_intercept && bias == NULL)
		l_bias = 1 / (n_features + 1) * normal_dist(m_prng);

	if(m_w)
		w = SGVector<float64_t>(m_w);
	else
	{
		w = SGVector<float64_t>(n_features);
		for (int i = 0; i < n_features; i++)
			w = 1 / (n_features + 1) * normal_dist(m_prng);
	}
	
	//Iterative Updates
	int n_iter = 0;
	for (int i = 0; i < m_max_iter; i++)
	{
		n_iter++;
		SGVector<float64_t> w_old(m_w);
		SGVector<float64_t> gradient_w = compute_grad_L2_loss_w(X, y, w, l_bias);
		float64_t gradient_bias = compute_grad_L2_loss_bias(X, y, w, l_bias);

		//Update
        w = linalg::add(w, gradient_w, 1.0, -1*m_learning_rate);
		if(m_fit_intercept)
			l_bias = l_bias - m_learning_rate * gradient_bias;
	
		//Apply proximal operator
		w = apply_proximal_operator(w, m_lambda * m_alpha);

		//Convergence by relative parameter change tolerance
		float64_t norm_update = linalg::norm(linalg::add(w, w_old, 1.0, -1.0));
		if(i > 0 && (norm_update / linalg::norm(w)) < m_tolerance)
			break;		
	}

	//Update the estimated variables
	m_w = w;
	bias = l_bias;

	return true;
}

//TODO
std::shared_ptr<RegressionLabels> GeneralizedLinearMachine::apply_regression(std::shared_ptr<Features> data)
{
	
}

//TODO
SGVector<float64_t> GeneralizedLinearMachine::apply_get_outputs(std::shared_ptr<Features> data)
{

}

//TODO
bool GeneralizedLinearMachine::train_machine(std::shared_ptr<const DenseFeatures<float64_t>> feats)
{

}

SGVector<float64_t> GeneralizedLinearMachine::conditional_intensity(SGMatrix<float64_t> X, SGVector<float64_t> w, float64_t l_bias)
{
	SGVector<float64_t> z = compute_z(X, w, l_bias);
	SGVector<float64_t> result = non_linearity(z);
	return result;
}

SGVector<float64_t> GeneralizedLinearMachine::compute_z(SGMatrix<float64_t> X, SGVector<float64_t> w, float64_t l_bias)
{
	SGVector<float64_t> z = linalg::matrix_prod(X, w, false);
	return z;
}

SGVector<float64_t> GeneralizedLinearMachine::non_linearity(SGVector<float64_t> z)
{
	SGVector<float64_t> result;
	switch (distribution)
	{
	case POISSON:
		result = SGVector<float64_t>(z);
		float64_t l_bias = 0;

		if(m_fit_intercept)
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
		break;
	}
	return result;
}

SGVector<float64_t> GeneralizedLinearMachine::gradient_non_linearity(SGVector<float64_t> z)
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
		break;
	}
	return result;
}

SGVector<float64_t> GeneralizedLinearMachine::compute_grad_L2_loss_w(SGMatrix<float64_t> X, SGVector<float64_t> y, SGVector<float64_t> w, float64_t l_bias)
{
	auto n_samples = y.vlen;
	auto n_features = X.num_rows;
	if(m_tau == NULL)
		m_tau = SGMatrix<float64_t>::create_identity_matrix(w.vlen, 1.0);
	SGMatrix<float64_t> inv_cov = linalg::matrix_prod(m_tau, m_tau, true, false);
	
	SGVector<float64_t> z = compute_z(X, w, l_bias);
	SGVector<float64_t> mu = non_linearity(z);
	SGVector<float64_t> grad_mu = gradient_non_linearity(z);

	SGVector<float64_t> grad_w(w.vlen);
	switch (distribution)
	{
	case POISSON:
		//PYTHON CODE
		//grad_w = ((np.dot(grad_mu.T, X) -
        //              np.dot((y * grad_mu / mu).T, X)).T)
		break;
	
	default:
		break;
	}
	for (int i = 0; i < grad_w.vlen; i++)
		grad_w[i] *= 1. / n_samples;
	
	linalg::add(grad_w, linalg::matrix_prod(inv_cov, w, false), 1.0, m_lambda * (1 - m_alpha));

	return grad_w;
}

float64_t GeneralizedLinearMachine::compute_grad_L2_loss_bias(SGMatrix<float64_t> X, SGVector<float64_t> y, SGVector<float64_t> w, float64_t l_bias)
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
		break;
	}
	grad_bias *= 1. / n_samples;

	return grad_bias;
}

SGVector<float64_t> GeneralizedLinearMachine::apply_proximal_operator(SGVector<float64_t> w, float64_t threshold)
{
	SGVector<float64_t> result(w);
	for (int i = 0; i < w.vlen; i++)
	{
		result[i] = 0;
		if(w[i]>0 && abs(w[i])>threshold)
			result[i] = abs(w[i]) - threshold;
		else if(w[i]>0 && abs(w[i])>threshold)
			result[i] = -1* (abs(w[i]) - threshold);
	}
	return result;
}