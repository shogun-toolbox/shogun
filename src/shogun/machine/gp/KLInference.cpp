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
#include <shogun/optimization/FirstOrderCostFunction.h>
#include <shogun/optimization/lbfgs/LBFGSMinimizer.h>


using namespace Eigen;

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
class KLInferenceCostFunction: public FirstOrderCostFunction
{
public:
        KLInferenceCostFunction():FirstOrderCostFunction() {  init(); }
        virtual ~KLInferenceCostFunction() { SG_UNREF(m_obj); }
        void set_target(CKLInference *obj)
        {
		REQUIRE(obj,"Obj must set\n");
		if(m_obj!=obj)
		{
			SG_REF(obj);
			SG_UNREF(m_obj);
			m_obj=obj;
		}
        }
        void unset_target(bool is_unref)
	{
		if(is_unref)
		{
			SG_UNREF(m_obj);
		}
		m_obj=NULL;
	}

        virtual float64_t get_cost()
        {
                REQUIRE(m_obj,"Object not set\n");
                bool status = m_obj->precompute();
                if (!status)
                        return CMath::NOT_A_NUMBER;
                float64_t nlml=m_obj->get_nlml_wrt_parameters();
                return nlml;
        
        }
        virtual SGVector<float64_t> obtain_variable_reference()
        {
                REQUIRE(m_obj,"Object not set\n");
                m_derivatives = SGVector<float64_t>((m_obj->m_alpha).vlen);
                return m_obj->m_alpha;
        }
        virtual SGVector<float64_t> get_gradient()
        {
                REQUIRE(m_obj,"Object not set\n");
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
			"derivatives in KLInferenceCostFunction", MS_NOT_AVAILABLE);
		SG_ADD((CSGObject **)&m_obj, "KLInferenceCostFunction__m_obj",
			"obj in KLInferenceCostFunction", MS_NOT_AVAILABLE);
        }
        CKLInference *m_obj;
};
#endif //DOXYGEN_SHOULD_SKIP_THIS

CKLInference::CKLInference() : CInference()
{
	init();
}

CKLInference::CKLInference(CKernel* kern,
		CFeatures* feat, CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod)
		: CInference(kern, feat, m, lab, mod)
{
	init();
	check_variational_likelihood(m_model);
}

void CKLInference::check_variational_likelihood(CLikelihoodModel* mod) const
{
	REQUIRE(mod, "the likelihood model must not be NULL\n")
	CVariationalGaussianLikelihood* lik= dynamic_cast<CVariationalGaussianLikelihood*>(mod);
	REQUIRE(lik,
		"The provided likelihood model (%s) must support variational Gaussian inference. ",
		"Please use a Variational Gaussian Likelihood model\n",
		mod->get_name());
}

void CKLInference::set_model(CLikelihoodModel* mod)
{
	check_variational_likelihood(mod);
	CInference::set_model(mod);
}

void CKLInference::init()
{
	m_noise_factor=1e-10;
	m_max_attempt=0;
	m_exp_factor=2;
	m_min_coeff_kernel=1e-5;
	SG_ADD(&m_noise_factor, "noise_factor",
		"The noise factor used for correcting Kernel matrix",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_exp_factor, "exp_factor",
		"The exponential factor used for increasing noise_factor",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_max_attempt, "max_attempt",
		"The max number of attempt to correct Kernel matrix",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_min_coeff_kernel, "min_coeff_kernel",
		"The minimum coeefficient of kernel matrix in LDLT factorization used to check whether the kernel matrix is positive definite or not",
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
	register_minimizer(new CLBFGSMinimizer());
}

CKLInference::~CKLInference()
{
}

void CKLInference::compute_gradient()
{
	CInference::compute_gradient();

	if (!m_gradient_update)
	{
		update_approx_cov();
		update_deriv();
		m_gradient_update=true;
		update_parameter_hash();
	}
}

void CKLInference::update()
{
	SG_DEBUG("entering\n");

	CInference::update();
	update_init();
	update_alpha();
	update_chol();
	m_gradient_update=false;
	update_parameter_hash();

	SG_DEBUG("leaving\n");
}

void CKLInference::set_noise_factor(float64_t noise_factor)
{
	REQUIRE(noise_factor>=0, "The noise_factor %.20f should be non-negative\n", noise_factor);
	m_noise_factor=noise_factor;
}

void CKLInference::set_min_coeff_kernel(float64_t min_coeff_kernel)
{
	REQUIRE(min_coeff_kernel>=0, "The min_coeff_kernel %.20f should be non-negative\n", min_coeff_kernel);
	m_min_coeff_kernel=min_coeff_kernel;
}

void CKLInference::set_max_attempt(index_t max_attempt)
{
	REQUIRE(max_attempt>=0, "The max_attempt %d should be non-negative. 0 means inifity attempts\n", max_attempt);
	m_max_attempt=max_attempt;
}

void CKLInference::set_exp_factor(float64_t exp_factor)
{
	REQUIRE(exp_factor>1.0, "The exp_factor %f should be greater than 1.0.\n", exp_factor);
	m_exp_factor=exp_factor;
}

void CKLInference::update_init()
{
	update_init_helper();
}

Eigen::LDLT<Eigen::MatrixXd> CKLInference::update_init_helper()
{
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	eigen_K=eigen_K+m_noise_factor*MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols);

	Eigen::LDLT<Eigen::MatrixXd> ldlt;
	ldlt.compute(eigen_K*CMath::exp(m_log_scale*2.0));

	float64_t attempt_count=0;
	MatrixXd Kernel_D=ldlt.vectorD();
	float64_t noise_factor=m_noise_factor;

	while (Kernel_D.minCoeff()<=m_min_coeff_kernel)
	{
		if (m_max_attempt>0 && attempt_count>m_max_attempt)
			SG_ERROR("The Kernel matrix is highly non-positive definite since the min_coeff_kernel is less than %.20f",
				" even when adding %.20f noise to the diagonal elements at max %d attempts\n",
				m_min_coeff_kernel, noise_factor, m_max_attempt);
		attempt_count++;
		float64_t pre_noise_factor=noise_factor;
		noise_factor*=m_exp_factor;
		//updat the noise  eigen_K=eigen_K+noise_factor*(m_exp_factor^attempt_count)*Identity()
		eigen_K=eigen_K+(noise_factor-pre_noise_factor)*MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols);
		ldlt.compute(eigen_K*CMath::exp(m_log_scale*2.0));
		Kernel_D=ldlt.vectorD();
	}

	return ldlt;
}


SGVector<float64_t> CKLInference::get_posterior_mean()
{
	compute_gradient();

	return SGVector<float64_t>(m_mu);
}

SGMatrix<float64_t> CKLInference::get_posterior_covariance()
{
	compute_gradient();

	return SGMatrix<float64_t>(m_Sigma);
}

CVariationalGaussianLikelihood* CKLInference::get_variational_likelihood() const
{
	check_variational_likelihood(m_model);
	CVariationalGaussianLikelihood* lik= dynamic_cast<CVariationalGaussianLikelihood*>(m_model);
	return lik;
}

float64_t CKLInference::get_nlml_wrt_parameters()
{
	CVariationalGaussianLikelihood * lik=get_variational_likelihood();
	lik->set_variational_distribution(m_mu, m_s2, m_labels);
	return get_negative_log_marginal_likelihood_helper();
}

float64_t CKLInference::get_negative_log_marginal_likelihood()
{
	if (parameter_hash_changed())
		update();

	return get_negative_log_marginal_likelihood_helper();
}

SGVector<float64_t> CKLInference::get_derivative_wrt_likelihood_model(const TParameter* param)
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

SGVector<float64_t> CKLInference::get_derivative_wrt_mean(const TParameter* param)
{
	// create eigen representation of K, Z, dfhat and alpha
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen/2);

	REQUIRE(param, "Param not set\n");
	SGVector<float64_t> result;
	int64_t len=const_cast<TParameter *>(param)->m_datatype.get_num_elements();
	result=SGVector<float64_t>(len);

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

float64_t CKLInference::optimization()
{
        KLInferenceCostFunction *cost_fun=new KLInferenceCostFunction();
        cost_fun->set_target(this);
	bool cleanup=false;
#ifdef USE_REFERENCE_COUNTING
	if(this->ref_count()>1)
		cleanup=true;
#endif
	FirstOrderMinimizer* opt= dynamic_cast<FirstOrderMinimizer*>(m_minimizer);

	REQUIRE(opt, "FirstOrderMinimizer is required\n")
	opt->set_cost_function(cost_fun);

        float64_t nlml_opt = opt->minimize();
	opt->unset_cost_function(false);
	cost_fun->unset_target(cleanup);

        SG_UNREF(cost_fun);
	return nlml_opt;
}

void CKLInference::register_minimizer(Minimizer* minimizer)
{
	REQUIRE(minimizer, "Minimizer must set\n");
	FirstOrderMinimizer* opt= dynamic_cast<FirstOrderMinimizer*>(minimizer);
	REQUIRE(opt, "FirstOrderMinimizer is required\n");
	CInference::register_minimizer(minimizer);
}


SGVector<float64_t> CKLInference::get_derivative_wrt_inference_method(const TParameter* param)
{
	REQUIRE(!strcmp(param->m_name, "log_scale"), "Can't compute derivative of "
			"the nagative log marginal likelihood wrt %s.%s parameter\n",
			get_name(), param->m_name)

	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	SGVector<float64_t> result(1);

	result[0]=get_derivative_related_cov(m_ktrtr);
	result[0]*=CMath::exp(m_log_scale*2.0)*2.0;

	return result;
}

SGVector<float64_t> CKLInference::get_derivative_wrt_kernel(const TParameter* param)
{
	REQUIRE(param, "Param not set\n");
	SGVector<float64_t> result;
	int64_t len=const_cast<TParameter *>(param)->m_datatype.get_num_elements();
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
		result[i]*=CMath::exp(m_log_scale*2.0);
	}

	return result;
}

SGMatrix<float64_t> CKLInference::get_cholesky()
{
	if (parameter_hash_changed())
		update();

	return SGMatrix<float64_t>(m_L);
}

} /* namespace shogun */

