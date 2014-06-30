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

#include <shogun/machine/gp/KLInferenceMethod.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/Math.h>

using namespace Eigen;

namespace shogun
{

CKLInferenceMethod::CKLInferenceMethod() : CInferenceMethod()
{
	init();
}

CKLInferenceMethod::CKLInferenceMethod(CKernel* kern,
		CFeatures* feat, CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod)
		: CInferenceMethod(kern, feat, m, lab, mod)
{
	init();
	check_variational_likelihood(m_model);
}

void CKLInferenceMethod::check_variational_likelihood(CLikelihoodModel* mod) const
{
	REQUIRE(mod, "the likelihood model must not be NULL\n")
	CVariationalGaussianLikelihood* lik= dynamic_cast<CVariationalGaussianLikelihood*>(mod);
	REQUIRE(lik,
		"The provided likelihood model (%s) must support variational Gaussian inference. ",
		"Please use a Variational Gaussian Likelihood model\n",
		mod->get_name());
}

void CKLInferenceMethod::set_model(CLikelihoodModel* mod)
{
	check_variational_likelihood(mod);
	CInferenceMethod::set_model(mod);
}

void CKLInferenceMethod::init()
{
	set_lbfgs_parameters();

	SG_ADD(&m_m, "m",
		"The number of corrections to approximate the inverse Hessian matrix",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_max_linesearch, "max_linesearch",
		"The maximum number of trials to do line search for each L-BFGS update",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_linesearch, "linesearch",
		"The line search algorithm",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_max_iterations, "max_iterations",
		"The maximum number of iterations for L-BFGS update",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_delta, "delta",
		"Delta for convergence test based on the change of function value",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_past, "past",
		"Distance for delta-based convergence test",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_epsilon, "epsilon",
		"Epsilon for convergence test based on the change of gradient",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_min_step, "min_step",
		"The minimum step of the line search",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_max_step, "max_step",
		"The maximum step of the line search",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_ftol, "ftol",
		"A parameter used in Armijo condition",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_wolfe, "wolfe",
		"A parameter used in curvature condition",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_gtol, "gtol",
		"A parameter used in Morethuente linesearch to control the accuracy",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_xtol, "xtol",
		"The machine precision for floating-point values",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_orthantwise_c, "orthantwise_c",
		"Coeefficient for the L1 norm of variables",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_orthantwise_start, "orthantwise_start",
		"Start index for computing L1 norm of the variables",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_orthantwise_end, "orthantwise_end",
		"End index for computing L1 norm of the variables",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_s2, "s2",
		"Variational parameter sigma2",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_mu, "mu",
		"Variational parameter mu and posterior mean",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_Sigma, "Sigma",
		"Posterior covariance matrix Sigma",
		MS_NOT_AVAILABLE);
}

CKLInferenceMethod::~CKLInferenceMethod()
{
}

void CKLInferenceMethod::update()
{
	SG_DEBUG("entering\n");

	CInferenceMethod::update();
	update_alpha();
	update_chol();
	update_approx_cov();
	update_deriv();
	update_parameter_hash();

	SG_DEBUG("leaving\n");
}

SGVector<float64_t> CKLInferenceMethod::get_posterior_mean()
{
	if (parameter_hash_changed())
		update();

	return SGVector<float64_t>(m_mu);
}

SGMatrix<float64_t> CKLInferenceMethod::get_posterior_covariance()
{
	if (parameter_hash_changed())
		update();

	return SGMatrix<float64_t>(m_Sigma);
}

float64_t CKLInferenceMethod::evaluate(void *obj, const float64_t *parameters,
	float64_t *gradient, const int dim, const float64_t step)
{
	/* Note that parameters = parameters_pre_iter - step * gradient_pre_iter */
	CKLInferenceMethod * obj_prt
		= static_cast<CKLInferenceMethod *>(obj);

	REQUIRE(obj_prt, "The instance object passed to L-BFGS optimizer should not be NULL\n");

	obj_prt->lbfgs_precompute();
	float64_t nlml=obj_prt->get_nlml_wrt_parameters();

	SGVector<float64_t> sg_gradient(gradient, dim, false);
	obj_prt->get_gradient_of_nlml_wrt_parameters(sg_gradient);

	return nlml;
}

CVariationalGaussianLikelihood* CKLInferenceMethod::get_variational_likelihood() const
{
	check_variational_likelihood(m_model);
	CVariationalGaussianLikelihood* lik= dynamic_cast<CVariationalGaussianLikelihood*>(m_model);
	return lik;
}

float64_t CKLInferenceMethod::get_nlml_wrt_parameters()
{
	CVariationalGaussianLikelihood * lik=get_variational_likelihood();
	lik->set_variational_distribution(m_mu, m_s2, m_labels);
	return get_negative_log_marginal_likelihood_helper();
}

void CKLInferenceMethod::set_lbfgs_parameters(
		int m,
		int max_linesearch,
		int linesearch,
		int max_iterations,
		float64_t delta,
		int past,
		float64_t epsilon,
		bool enable_newton_if_fail,
		float64_t min_step,
		float64_t max_step,
		float64_t ftol,
		float64_t wolfe,
		float64_t gtol,
		float64_t xtol,
		float64_t orthantwise_c,
		int orthantwise_start,
		int orthantwise_end)
{
	m_m = m;
	m_max_linesearch = max_linesearch;
	m_linesearch = linesearch;
	m_max_iterations = max_iterations;
	m_delta = delta;
	m_past = past;
	m_epsilon = epsilon;
	m_min_step = min_step;
	m_max_step = max_step;
	m_ftol = ftol;
	m_wolfe = wolfe;
	m_gtol = gtol;
	m_xtol = xtol;
	m_orthantwise_c = orthantwise_c;
	m_orthantwise_start = orthantwise_start;
	m_orthantwise_end = orthantwise_end;
}

float64_t CKLInferenceMethod::get_negative_log_marginal_likelihood()
{
	if (parameter_hash_changed())
		update();
	
	return get_negative_log_marginal_likelihood_helper();
}

SGVector<float64_t> CKLInferenceMethod::get_derivative_wrt_likelihood_model(const TParameter* param)
{
	CVariationalLikelihood * lik=get_variational_likelihood();
	if (!lik->supports_derivative_wrt_hyperparameter())
		return SGVector<float64_t> ();

	//%lp_dhyp = likKL(v,lik,hyp.lik,y,K*post.alpha+m,[],[],j);
	SGVector<float64_t> lp_dhyp=lik->get_first_derivative_wrt_hyperparameter(param);
	Map<VectorXd> eigen_lp_dhyp(lp_dhyp.vector, lp_dhyp.vlen);
	SGVector<float64_t> result(1);
	//%dnlZ.lik(j) = -sum(lp_dhyp);
	result[0]=-eigen_lp_dhyp.sum();

	return result;
}

SGVector<float64_t> CKLInferenceMethod::get_derivative_wrt_mean(const TParameter* param)
{
	// create eigen representation of K, Z, dfhat and alpha
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen/2);

	SGVector<float64_t> result;

	if (param->m_datatype.m_ctype==CT_VECTOR ||
			param->m_datatype.m_ctype==CT_SGVECTOR)
	{
		REQUIRE(param->m_datatype.m_length_y,
				"Length of the parameter %s should not be NULL\n", param->m_name)

		result=SGVector<float64_t>(*(param->m_datatype.m_length_y));
	}
	else
	{
		result=SGVector<float64_t>(1);
	}

	for (index_t i=0; i<result.vlen; i++)
	{
		SGVector<float64_t> dmu;

		//%dm_t = feval(mean{:}, hyp.mean, x, j);
		if (result.vlen==1)
			dmu=m_mean->get_parameter_derivative(m_features, param);
		else
			dmu=m_mean->get_parameter_derivative(m_features, param, i);

		Map<VectorXd> eigen_dmu(dmu.vector, dmu.vlen);

		//%dnlZ.mean(j) = -alpha'*dm_t;
		result[i]=-eigen_alpha.dot(eigen_dmu);
	}

	return result;
}

float64_t CKLInferenceMethod::lbfgs_optimization()
{
	lbfgs_parameter_t lbfgs_param;
	lbfgs_param.m = m_m;
	lbfgs_param.max_linesearch = m_max_linesearch;
	lbfgs_param.linesearch = m_linesearch;
	lbfgs_param.max_iterations = m_max_iterations;
	lbfgs_param.delta = m_delta;
	lbfgs_param.past = m_past;
	lbfgs_param.epsilon = m_epsilon;
	lbfgs_param.min_step = m_min_step;
	lbfgs_param.max_step = m_max_step;
	lbfgs_param.ftol = m_ftol;
	lbfgs_param.wolfe = m_wolfe;
	lbfgs_param.gtol = m_gtol;
	lbfgs_param.xtol = m_xtol;
	lbfgs_param.orthantwise_c = m_orthantwise_c;
	lbfgs_param.orthantwise_start = m_orthantwise_start;
	lbfgs_param.orthantwise_end = m_orthantwise_end;

	float64_t nlml_opt=0;
	void * obj_prt = static_cast<void *>(this);
	lbfgs(m_alpha.vlen, m_alpha.vector, &nlml_opt,
		CKLInferenceMethod::evaluate,
		NULL, obj_prt, &lbfgs_param);

	return nlml_opt;
}

SGVector<float64_t> CKLInferenceMethod::get_derivative_wrt_inference_method(const TParameter* param)
{
	REQUIRE(!strcmp(param->m_name, "scale"), "Can't compute derivative of "
			"the nagative log marginal likelihood wrt %s.%s parameter\n",
			get_name(), param->m_name)

	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	// compute derivative K wrt scale
	MatrixXd eigen_dK=eigen_K*m_scale*2.0;

	SGVector<float64_t> result(1);

	result[0]=get_derivative_related_cov(eigen_dK);

	return result;
}

SGVector<float64_t> CKLInferenceMethod::get_derivative_wrt_kernel(const TParameter* param)
{
	SGVector<float64_t> result;

	if (param->m_datatype.m_ctype==CT_VECTOR ||
			param->m_datatype.m_ctype==CT_SGVECTOR)
	{
		REQUIRE(param->m_datatype.m_length_y,
				"Length of the parameter %s should not be NULL\n", param->m_name)
		result=SGVector<float64_t>(*(param->m_datatype.m_length_y));
	}
	else
	{
		result=SGVector<float64_t>(1);
	}

	for (index_t i=0; i<result.vlen; i++)
	{
		SGMatrix<float64_t> dK(m_mu.vlen, m_mu.vlen);

		//dK = feval(covfunc{:},hyper,x,j);
		if (result.vlen==1)
			dK=m_kernel->get_parameter_gradient(param);
		else
			dK=m_kernel->get_parameter_gradient(param, i);

		Map<MatrixXd> eigen_dK(dK.matrix, dK.num_cols, dK.num_rows);

		result[i]=get_derivative_related_cov(eigen_dK*CMath::sq(m_scale));
	}

	return result;
}

SGMatrix<float64_t> CKLInferenceMethod::get_cholesky()
{
	if (parameter_hash_changed())
		update();

	return SGMatrix<float64_t>(m_L);
}

} /* namespace shogun */

#endif /* HAVE_EIGEN3 */
