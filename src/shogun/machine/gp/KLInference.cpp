 /*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 * This code specifically adapted from function in approxKL.m and infKL.m
 */

#include <shogun/machine/gp/KLInference.h>

#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/visitors/ShapeVisitor.h>
#include <shogun/optimization/FirstOrderCostFunction.h>
#include <shogun/optimization/lbfgs/LBFGSMinimizer.h>

#include <utility>


using namespace Eigen;

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
class KLInferenceCostFunction: public FirstOrderCostFunction
{
public:
        KLInferenceCostFunction():FirstOrderCostFunction() {  init(); }
        virtual ~KLInferenceCostFunction() {  }
        void set_target(const std::shared_ptr<KLInference>&obj)
        {
			require(obj,"Obj must set");
			m_obj=obj;
	    }

        virtual float64_t get_cost()
        {
                require(m_obj,"Object not set");
                bool status = m_obj->precompute();
                if (!status)
                        return Math::NOT_A_NUMBER;
                float64_t nlml=m_obj->get_nlml_wrt_parameters();
                return nlml;

        }
        virtual SGVector<float64_t> obtain_variable_reference()
        {
                require(m_obj,"Object not set");
                m_derivatives = SGVector<float64_t>((m_obj->m_alpha).vlen);
                return m_obj->m_alpha;
        }
        virtual SGVector<float64_t> get_gradient()
        {
                require(m_obj,"Object not set");
		m_obj->get_gradient_of_nlml_wrt_parameters(m_derivatives);
                return m_derivatives;
        }

	virtual const char* get_name() const { return "KLInferenceCostFunction"; }
private:
        SGVector<float64_t> m_derivatives;
        void init()
        {
                m_obj=NULL;
                m_derivatives = SGVector<float64_t>();
		SG_ADD(&m_derivatives, "KLInferenceCostFunction__m_derivatives",
			"derivatives in KLInferenceCostFunction");
		SG_ADD((std::shared_ptr<SGObject>*)&m_obj, "KLInferenceCostFunction__m_obj",
			"obj in KLInferenceCostFunction");
        }
        std::shared_ptr<KLInference> m_obj;
};
#endif //DOXYGEN_SHOULD_SKIP_THIS

KLInference::KLInference() : Inference()
{
	init();
}

KLInference::KLInference(std::shared_ptr<Kernel> kern,
		std::shared_ptr<Features> feat, std::shared_ptr<MeanFunction> m, std::shared_ptr<Labels> lab, std::shared_ptr<LikelihoodModel> mod)
		: Inference(std::move(kern), std::move(feat), std::move(m), std::move(lab), std::move(mod))
{
	init();
	check_variational_likelihood(m_model);
}

void KLInference::check_variational_likelihood(std::shared_ptr<LikelihoodModel> mod) const
{
	require(mod, "the likelihood model must not be NULL");
	auto lik= std::dynamic_pointer_cast<VariationalGaussianLikelihood>(mod);
	require(lik,
		"The provided likelihood model ({}) must support variational Gaussian inference. ",
		"Please use a Variational Gaussian Likelihood model",
		mod->get_name());
}

void KLInference::set_model(std::shared_ptr<LikelihoodModel> mod)
{
	check_variational_likelihood(mod);
	Inference::set_model(mod);
}

void KLInference::init()
{
	m_noise_factor=1e-10;
	m_max_attempt=0;
	m_exp_factor=2;
	m_min_coeff_kernel=1e-5;
	SG_ADD(&m_noise_factor, "noise_factor",
		"The noise factor used for correcting Kernel matrix");
	SG_ADD(&m_exp_factor, "exp_factor",
		"The exponential factor used for increasing noise_factor");
	SG_ADD(&m_max_attempt, "max_attempt",
		"The max number of attempt to correct Kernel matrix");
	SG_ADD(&m_min_coeff_kernel, "min_coeff_kernel",
		"The minimum coeefficient of kernel matrix in LDLT factorization used to check whether the kernel matrix is positive definite or not");
	SG_ADD(&m_s2, "s2",
		"Variational parameter sigma2");
	SG_ADD(&m_mu, "mu",
		"Variational parameter mu and posterior mean");
	SG_ADD(&m_Sigma, "Sigma",
		"Posterior covariance matrix Sigma");
	register_minimizer(std::make_shared<LBFGSMinimizer>());
}

KLInference::~KLInference()
{
}

void KLInference::compute_gradient()
{
	Inference::compute_gradient();

	if (!m_gradient_update)
	{
		update_approx_cov();
		update_deriv();
		m_gradient_update=true;
		update_parameter_hash();
	}
}

void KLInference::update()
{
	SG_TRACE("entering");

	Inference::update();
	update_init();
	update_alpha();
	update_chol();
	m_gradient_update=false;
	update_parameter_hash();

	SG_TRACE("leaving");
}

void KLInference::set_noise_factor(float64_t noise_factor)
{
	require(noise_factor>=0, "The noise_factor {:.20f} should be non-negative", noise_factor);
	m_noise_factor=noise_factor;
}

void KLInference::set_min_coeff_kernel(float64_t min_coeff_kernel)
{
	require(min_coeff_kernel>=0, "The min_coeff_kernel {:.20f} should be non-negative", min_coeff_kernel);
	m_min_coeff_kernel=min_coeff_kernel;
}

void KLInference::set_max_attempt(index_t max_attempt)
{
	require(max_attempt>=0, "The max_attempt {} should be non-negative. 0 means inifity attempts", max_attempt);
	m_max_attempt=max_attempt;
}

void KLInference::set_exp_factor(float64_t exp_factor)
{
	require(exp_factor>1.0, "The exp_factor {} should be greater than 1.0.", exp_factor);
	m_exp_factor=exp_factor;
}

void KLInference::update_init()
{
	update_init_helper();
}

Eigen::LDLT<Eigen::MatrixXd> KLInference::update_init_helper()
{
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	eigen_K=eigen_K+m_noise_factor*MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols);

	Eigen::LDLT<Eigen::MatrixXd> ldlt;
	ldlt.compute(eigen_K * std::exp(m_log_scale * 2.0));

	float64_t attempt_count=0;
	MatrixXd Kernel_D=ldlt.vectorD();
	float64_t noise_factor=m_noise_factor;

	while (Kernel_D.minCoeff()<=m_min_coeff_kernel)
	{
		if (m_max_attempt>0 && attempt_count>m_max_attempt)
			error("The Kernel matrix is highly non-positive definite since the min_coeff_kernel is less than {:.20f}",
				" even when adding {:.20f} noise to the diagonal elements at max {} attempts",
				m_min_coeff_kernel, noise_factor, m_max_attempt);
		attempt_count++;
		float64_t pre_noise_factor=noise_factor;
		noise_factor*=m_exp_factor;
		//updat the noise  eigen_K=eigen_K+noise_factor*(m_exp_factor^attempt_count)*Identity()
		eigen_K=eigen_K+(noise_factor-pre_noise_factor)*MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols);
		ldlt.compute(eigen_K * std::exp(m_log_scale * 2.0));
		Kernel_D=ldlt.vectorD();
	}

	return ldlt;
}


SGVector<float64_t> KLInference::get_posterior_mean()
{
	compute_gradient();

	return SGVector<float64_t>(m_mu);
}

SGMatrix<float64_t> KLInference::get_posterior_covariance()
{
	compute_gradient();

	return SGMatrix<float64_t>(m_Sigma);
}

std::shared_ptr<VariationalGaussianLikelihood> KLInference::get_variational_likelihood() const
{
	check_variational_likelihood(m_model);
	auto lik= std::dynamic_pointer_cast<VariationalGaussianLikelihood>(m_model);
	return lik;
}

float64_t KLInference::get_nlml_wrt_parameters()
{
	auto lik=get_variational_likelihood();
	lik->set_variational_distribution(m_mu, m_s2, m_labels);
	return get_negative_log_marginal_likelihood_helper();
}

float64_t KLInference::get_negative_log_marginal_likelihood()
{
	if (parameter_hash_changed())
		update();

	return get_negative_log_marginal_likelihood_helper();
}

SGVector<float64_t> KLInference::get_derivative_wrt_likelihood_model(Parameters::const_reference param)
{
	auto lik=get_variational_likelihood();
	if (!lik->supports_derivative_wrt_hyperparameter())
		return SGVector<float64_t> ();

	//%lp_dhyp = likKL(v,lik,hyp.lik,y,K*post.alpha+m,[],[],j);
	SGVector<float64_t> lp_dhyp=lik->get_first_derivative_wrt_hyperparameter(param);
	Map<VectorXd> eigen_lp_dhyp(lp_dhyp.vector, lp_dhyp.vlen);
	SGVector<float64_t> result(1);
	//{}nlZ.lik(j) = -sum(lp_dhyp);
	result[0]=-eigen_lp_dhyp.sum();

	return result;
}

SGVector<float64_t> KLInference::get_derivative_wrt_mean(Parameters::const_reference param)
{
	// create eigen representation of K, Z, dfhat and alpha
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen/2);

	SGVector<float64_t> result;
	auto visitor = std::make_unique<ShapeVisitor>();
	param.second->get_value().visit(visitor.get());
	int64_t len= visitor->get_size();
	result=SGVector<float64_t>(len);

	for (index_t i=0; i<result.vlen; i++)
	{
		SGVector<float64_t> dmu;

		//{}m_t = feval(mean{:}, hyp.mean, x, j);
		if (result.vlen==1)
			dmu=m_mean->get_parameter_derivative(m_features, param);
		else
			dmu=m_mean->get_parameter_derivative(m_features, param, i);

		Map<VectorXd> eigen_dmu(dmu.vector, dmu.vlen);

		//{}nlZ.mean(j) = -alpha'*dm_t;
		result[i]=-eigen_alpha.dot(eigen_dmu);
	}

	return result;
}

float64_t KLInference::optimization()
{
    auto cost_fun=std::make_shared<KLInferenceCostFunction>();
    cost_fun->set_target(shared_from_this()->as<KLInference>());

	auto opt=m_minimizer->as<FirstOrderMinimizer>();

	require(opt, "FirstOrderMinimizer is required");
	opt->set_cost_function(cost_fun);

	float64_t nlml_opt = opt->minimize();
	opt->unset_cost_function(false);

	return nlml_opt;
}

void KLInference::register_minimizer(std::shared_ptr<Minimizer> minimizer)
{
	require(minimizer, "Minimizer must set");
	auto opt= std::dynamic_pointer_cast<FirstOrderMinimizer>(minimizer);
	require(opt, "FirstOrderMinimizer is required");
	Inference::register_minimizer(minimizer);
}


SGVector<float64_t> KLInference::get_derivative_wrt_inference_method(Parameters::const_reference param)
{
	require(param.first == "log_scale", "Can't compute derivative of "
			"the nagative log marginal likelihood wrt {}.{} parameter",
			get_name(), param.first);

	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	SGVector<float64_t> result(1);

	result[0]=get_derivative_related_cov(m_ktrtr);
	result[0] *= std::exp(m_log_scale * 2.0) * 2.0;

	return result;
}

SGVector<float64_t> KLInference::get_derivative_wrt_kernel(Parameters::const_reference param)
{
	SGVector<float64_t> result;
	auto visitor = std::make_unique<ShapeVisitor>();
	param.second->get_value().visit(visitor.get());
	int64_t len= visitor->get_size();
	result=SGVector<float64_t>(len);

	for (index_t i=0; i<result.vlen; i++)
	{
		SGMatrix<float64_t> dK(m_mu.vlen, m_mu.vlen);

		//dK = feval(covfunc{:},hyper,x,j);
		if (result.vlen==1)
			dK=m_kernel->get_parameter_gradient(param);
		else
			dK=m_kernel->get_parameter_gradient(param, i);

		result[i]=get_derivative_related_cov(dK);
		result[i] *= std::exp(m_log_scale * 2.0);
	}

	return result;
}

SGMatrix<float64_t> KLInference::get_cholesky()
{
	if (parameter_hash_changed())
		update();

	return SGMatrix<float64_t>(m_L);
}

} /* namespace shogun */

