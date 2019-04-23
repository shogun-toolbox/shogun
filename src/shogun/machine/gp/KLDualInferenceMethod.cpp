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
 * Code adapted from
 * http://hannes.nickisch.org/code/approxXX.tar.gz
 * and Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and the reference paper is
 * Mohammad Emtiyaz Khan, Aleksandr Y. Aravkin, Michael P. Friedlander, Matthias Seeger
 * Fast Dual Variational Inference for Non-Conjugate Latent Gaussian Models. ICML2013*
 *
 * This code specifically adapted from function in approxKL.m and infKL.m
 */

#include <shogun/machine/gp/KLDualInferenceMethod.h>

#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/gp/MatrixOperations.h>
#include <shogun/machine/gp/DualVariationalGaussianLikelihood.h>
#include <shogun/labels/BinaryLabels.h>

using namespace Eigen;

namespace shogun
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
class KLDualInferenceMethodCostFunction: public FirstOrderCostFunction
{
friend class KLDualInferenceMethodMinimizer;
public:
	KLDualInferenceMethodCostFunction():FirstOrderCostFunction() {  init(); }
	virtual ~KLDualInferenceMethodCostFunction() {  }
	void set_target(std::shared_ptr<KLDualInferenceMethod >obj)
	{
		REQUIRE(obj, "Obj must set\n");
		if(m_obj != obj)
		{


			m_obj=obj;
		}
	}
	void unset_target(bool is_unref)
	{
		if(is_unref)
		{

		}
		m_obj=NULL;
	}
	virtual float64_t get_cost()
	{
		REQUIRE(m_obj,"Object not set\n");
		bool status=m_obj->precompute();
		if (status)
		{
			float64_t nlml=m_obj->get_dual_objective_wrt_parameters();
			return nlml;
		}
		return Math::NOT_A_NUMBER;
	}
	virtual SGVector<float64_t> obtain_variable_reference()
	{
		REQUIRE(m_obj,"Object not set\n");
		m_derivatives = SGVector<float64_t>((m_obj->m_W).vlen);
		return m_obj->m_W;
	}
	virtual SGVector<float64_t> get_gradient()
	{
		REQUIRE(m_obj,"Object not set\n");
		m_obj->get_gradient_of_dual_objective_wrt_parameters(m_derivatives);
		return m_derivatives;
	}
	virtual const char* get_name() const { return "KLDualInferenceMethodCostFunction"; }
private:
	SGVector<float64_t> m_derivatives;
	void init()
	{
		m_obj=NULL;
		m_derivatives = SGVector<float64_t>();
		SG_ADD(&m_derivatives, "KLDualInferenceMethodCostFunction__m_derivatives",
			"derivatives in KLDualInferenceMethodCostFunction");
		SG_ADD((std::shared_ptr<SGObject>*)&m_obj, "KLDualInferenceMethodCostFunction__m_obj",
			"obj in KLDualInferenceMethodCostFunction");
	}
	std::shared_ptr<KLDualInferenceMethod >m_obj;
	std::shared_ptr<DualVariationalGaussianLikelihood> get_dual_variational_likelihood() const
	{
		REQUIRE(m_obj,"Object not set\n");
		return m_obj->get_dual_variational_likelihood();
	}
};
#endif //DOXYGEN_SHOULD_SKIP_THIS

void KLDualInferenceMethodMinimizer::init_minimization()
{
	ELBFGSLineSearch linesearch=LBFGSLineSearchHelper::get_lbfgs_linear_search(m_linesearch_id);
	REQUIRE((linesearch == BACKTRACKING_ARMIJO) ||
		(linesearch == BACKTRACKING_WOLFE) ||
		(linesearch == BACKTRACKING_STRONG_WOLFE),
		"The provided line search method is not supported. Please use backtracking line search methods\n");
	LBFGSMinimizer::init_minimization();
}

float64_t KLDualInferenceMethodMinimizer::minimize()
{
	lbfgs_parameter_t lbfgs_param;
	lbfgs_param.m = m_m;
	lbfgs_param.max_linesearch = m_max_linesearch;
	lbfgs_param.linesearch = LBFGSLineSearchHelper::get_lbfgs_linear_search(m_linesearch_id);
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

	init_minimization();

	float64_t cost=0.0;
	int error_code=lbfgs(m_target_variable.vlen, m_target_variable.vector,
		&cost, KLDualInferenceMethodMinimizer::evaluate,
		NULL, this, &lbfgs_param, KLDualInferenceMethodMinimizer::adjust_step);

	if(error_code!=0 && error_code!=LBFGS_ALREADY_MINIMIZED)
	{
	  SG_SWARNING("Error(s) happened during L-BFGS optimization (error code:%d)\n",
		  error_code);
	}
	return cost;
}

float64_t KLDualInferenceMethodMinimizer::evaluate(void *obj, const float64_t *variable,
	float64_t *gradient, const int dim, const float64_t step)
{
	/* Note that parameters = parameters_pre_iter - step * gradient_pre_iter */
	auto obj_prt
		= (KLDualInferenceMethodMinimizer*)obj;

	REQUIRE(obj_prt, "The instance object passed to L-BFGS optimizer should not be NULL\n");

	float64_t cost=obj_prt->m_fun->get_cost();
	if (Math::is_nan(cost) || std::isinf(cost))
			return cost;
	//get the gradient wrt variable_new
	SGVector<float64_t> grad=obj_prt->m_fun->get_gradient();
	REQUIRE(grad.vlen==dim,
		"The length of gradient (%d) and the length of variable (%d) do not match\n",
		grad.vlen,dim);

	std::copy(grad.vector,grad.vector+dim,gradient);
	return cost;
}

float64_t KLDualInferenceMethodMinimizer::adjust_step(void *obj, const float64_t *parameters,
	const float64_t *direction, const int dim, const float64_t step)
{
	/* Note that parameters = parameters_pre_iter - step * gradient_pre_iter */
	auto obj_prt
		= (KLDualInferenceMethodMinimizer*)obj;

	REQUIRE(obj_prt, "The instance object passed to L-BFGS optimizer should not be NULL\n");

	float64_t *non_const_direction=const_cast<float64_t *>(direction);
	SGVector<float64_t> sg_direction(non_const_direction, dim, false);

	auto fun=std::dynamic_pointer_cast<KLDualInferenceMethodCostFunction>(obj_prt->m_fun);
	REQUIRE(fun, "The cost function must be KLDualInferenceMethodCostFunction\n");

	auto lik=fun->get_dual_variational_likelihood();

	float64_t adjust_stp=lik->adjust_step_wrt_dual_parameter(sg_direction, step);
	return adjust_stp;
}

KLDualInferenceMethod::KLDualInferenceMethod() : KLInference()
{
	init();
}

KLDualInferenceMethod::KLDualInferenceMethod(std::shared_ptr<Kernel> kern,
		std::shared_ptr<Features> feat, std::shared_ptr<MeanFunction> m, std::shared_ptr<Labels> lab, std::shared_ptr<LikelihoodModel> mod)
		: KLInference(kern, feat, m, lab, mod)
{
	init();
}

std::shared_ptr<KLDualInferenceMethod> KLDualInferenceMethod::obtain_from_generic(
		std::shared_ptr<Inference> inference)
{
	if (inference==NULL)
		return NULL;

	if (inference->get_inference_type()!=INF_KL_DUAL)
	{
		SG_SERROR("Provided inference is not of type CKLDualInferenceMethod!\n");
	}


	return inference->as<KLDualInferenceMethod>();
}

SGVector<float64_t> KLDualInferenceMethod::get_alpha()
{
	if (parameter_hash_changed())
		update();

	SGVector<float64_t> result(m_alpha);
	return result;
}

KLDualInferenceMethod::~KLDualInferenceMethod()
{
}

void KLDualInferenceMethod::check_dual_inference(std::shared_ptr<LikelihoodModel> mod) const
{
	auto lik=mod->as<DualVariationalGaussianLikelihood>();
	REQUIRE(lik,
		"The provided likelihood model is not a variational dual Likelihood model.\n");
}

void KLDualInferenceMethod::set_model(std::shared_ptr<LikelihoodModel> mod)
{
	check_dual_inference(mod);
	KLInference::set_model(mod);
}

std::shared_ptr<DualVariationalGaussianLikelihood> KLDualInferenceMethod::get_dual_variational_likelihood() const
{
	check_dual_inference(m_model);
	return m_model->as<DualVariationalGaussianLikelihood>();
}

void KLDualInferenceMethod::register_minimizer(std::shared_ptr<Minimizer> minimizer)
{
	auto opt=minimizer->as<KLDualInferenceMethodMinimizer>();
	REQUIRE(opt,"The minimizer must be an instance of KLDualInferenceMethodMinimizer\n");
	Inference::register_minimizer(minimizer);
}


void KLDualInferenceMethod::init()
{
	SG_ADD(&m_W, "W",
		"noise matrix W");
	SG_ADD(&m_sW, "sW",
		"Square root of noise matrix W");
	SG_ADD(&m_dv, "dv",
		"the gradient of the variational expection wrt sigma2");
	SG_ADD(&m_df, "df",
		"the gradient of the variational expection wrt mu");
	SG_ADD(&m_is_dual_valid, "is_dual_valid",
		"whether the lambda (m_W) is valid or not");

	m_is_dual_valid=false;
	register_minimizer(std::make_shared<KLDualInferenceMethodMinimizer>());
}

bool KLDualInferenceMethod::precompute()
{
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	auto lik= get_dual_variational_likelihood();
	Map<VectorXd> eigen_W(m_W.vector, m_W.vlen);

	lik->set_dual_parameters(m_W, m_labels);
	m_is_dual_valid=lik->dual_parameters_valid();

	if (!m_is_dual_valid)
		return false;

	//construct alpha
	m_alpha=lik->get_mu_dual_parameter();
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	eigen_alpha=-eigen_alpha;

	Map<VectorXd> eigen_sW(m_sW.vector, m_sW.vlen);
	eigen_sW=eigen_W.array().sqrt().matrix();

	m_L = CMatrixOperations::get_choleksy(
		m_W, m_sW, m_ktrtr, std::exp(m_log_scale));
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	//solve L'*V=diag(sW)*K
	Map<MatrixXd> eigen_V(m_V.matrix, m_V.num_rows, m_V.num_cols);
	eigen_V = eigen_L.triangularView<Upper>().adjoint().solve(
		eigen_sW.asDiagonal() * eigen_K * std::exp(m_log_scale * 2.0));
	Map<VectorXd> eigen_s2(m_s2.vector, m_s2.vlen);
	//Sigma=inv(inv(K)+diag(W))=K-K*diag(sW)*inv(L)'*inv(L)*diag(sW)*K
	//v=abs(diag(Sigma))
	eigen_s2 = (eigen_K.diagonal().array() * std::exp(m_log_scale * 2.0) -
		        (eigen_V.array().pow(2).colwise().sum().transpose()))
		           .abs()
		           .matrix();

	//construct mu
	SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);
	Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	//mu=K*alpha+m
	eigen_mu = eigen_K * std::exp(m_log_scale * 2.0) * eigen_alpha + eigen_mean;
	return true;
}

float64_t KLDualInferenceMethod::get_dual_objective_wrt_parameters()
{
	if (!m_is_dual_valid)
		return Math::INFTY;

	SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);
	Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	auto lik= get_dual_variational_likelihood();

	float64_t a=SGVector<float64_t>::sum(lik->get_dual_objective_value());
	float64_t result=0.5*eigen_alpha.dot(eigen_mu-eigen_mean)+a;
	result+=eigen_mean.dot(eigen_alpha);
	result-=eigen_L.diagonal().array().log().sum();

	return result;
}

void KLDualInferenceMethod::get_gradient_of_dual_objective_wrt_parameters(SGVector<float64_t> gradient)
{
	REQUIRE(gradient.vlen==m_alpha.vlen,
		"The length of gradients (%d) should the same as the length of parameters (%d)\n",
		gradient.vlen, m_alpha.vlen);

	if (!m_is_dual_valid)
		return;

	Map<VectorXd> eigen_gradient(gradient.vector, gradient.vlen);

	auto lik= get_dual_variational_likelihood();

	TParameter* lambda_param=lik->m_parameters->get_parameter("lambda");
	SGVector<float64_t>d_lambda=lik->get_dual_first_derivative(lambda_param);
	Map<VectorXd> eigen_d_lambda(d_lambda.vector, d_lambda.vlen);

	Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	Map<VectorXd> eigen_s2(m_s2.vector, m_s2.vlen);
	eigen_gradient=-eigen_mu-0.5*eigen_s2+eigen_d_lambda;
}

float64_t KLDualInferenceMethod::get_nlml_wrapper(SGVector<float64_t> alpha, SGVector<float64_t> mu, SGMatrix<float64_t> L)
{
	Map<MatrixXd> eigen_L(L.matrix, L.num_rows, L.num_cols);
	Map<VectorXd> eigen_alpha(alpha.vector, alpha.vlen);
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	//get mean vector and create eigen representation of it
	SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mu(mu.vector, mu.vlen);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);

	auto lik=get_dual_variational_likelihood();

	SGVector<float64_t>lab=m_labels->as<BinaryLabels>()->get_labels();
	Map<VectorXd> eigen_lab(lab.vector, lab.vlen);

	float64_t a=SGVector<float64_t>::sum(lik->get_variational_expection());

	float64_t trace=0;
	//L_inv=L\eye(n);
	//trace(L_inv'*L_inv)   %V*inv(K)
	MatrixXd eigen_t=eigen_L.triangularView<Upper>().adjoint().solve(MatrixXd::Identity(eigen_L.rows(),eigen_L.cols()));

	for(index_t idx=0; idx<eigen_t.rows(); idx++)
		trace +=(eigen_t.col(idx).array().pow(2)).sum();

	//nlZ = -a -logdet(V*inv(K))/2 -n/2 +(alpha'*K*alpha)/2 +trace(V*inv(K))/2;
	float64_t result=-a+eigen_L.diagonal().array().log().sum();

	result+=0.5*(-eigen_K.rows()+eigen_alpha.dot(eigen_mu-eigen_mean)+trace);
	return result;
}

float64_t KLDualInferenceMethod::get_negative_log_marginal_likelihood_helper()
{
	auto lik=get_dual_variational_likelihood();
	bool status = lik->set_variational_distribution(m_mu, m_s2, m_labels);
	if (status)
		return get_nlml_wrapper(m_alpha, m_mu, m_L);
	return Math::NOT_A_NUMBER;
}

float64_t KLDualInferenceMethod::get_derivative_related_cov(SGMatrix<float64_t> dK)
{
	Map<MatrixXd> eigen_dK(dK.matrix, dK.num_rows, dK.num_cols);
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> eigen_W(m_W.vector, m_W.vlen);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<VectorXd> eigen_sW(m_sW.vector, m_sW.vlen);
	Map<MatrixXd> eigen_Sigma(m_Sigma.matrix, m_Sigma.num_rows, m_Sigma.num_cols);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	Map<VectorXd> eigen_dv(m_dv.vector, m_dv.vlen);
	Map<VectorXd> eigen_df(m_df.vector, m_df.vlen);

	index_t len=m_W.vlen;
	//U=inv(L')*diag(sW)
	MatrixXd eigen_U=eigen_L.triangularView<Upper>().adjoint().solve(MatrixXd(eigen_sW.asDiagonal()));
	//A=I-K*diag(sW)*inv(L)*inv(L')*diag(sW)
	Map<MatrixXd> eigen_V(m_V.matrix, m_V.num_rows, m_V.num_cols);
	MatrixXd eigen_A=MatrixXd::Identity(len, len)-eigen_V.transpose()*eigen_U;

	//AdK = A*dK;
	MatrixXd AdK=eigen_A*eigen_dK;

	//z = diag(AdK) + sum(A.*AdK,2) - sum(A'.*AdK,1)';
	VectorXd z=AdK.diagonal()+(eigen_A.array()*AdK.array()).rowwise().sum().matrix()
		-(eigen_A.transpose().array()*AdK.array()).colwise().sum().transpose().matrix();

	float64_t result=eigen_alpha.dot(eigen_dK*(eigen_alpha/2.0-eigen_df))-z.dot(eigen_dv);

	return result;
}

void KLDualInferenceMethod::update_alpha()
{
	float64_t nlml_new=0;
	float64_t nlml_def=0;

	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	auto lik= get_dual_variational_likelihood();

	if (m_alpha.vlen == m_labels->get_num_labels())
	{
		nlml_new=get_negative_log_marginal_likelihood_helper();
		index_t len=m_labels->get_num_labels();
		SGVector<float64_t> W_tmp(len);
		Map<VectorXd> eigen_W(W_tmp.vector, W_tmp.vlen);
		eigen_W.fill(0.5);
		SGVector<float64_t> sW_tmp(len);
		Map<VectorXd> eigen_sW(sW_tmp.vector, sW_tmp.vlen);
		eigen_sW=eigen_W.array().sqrt().matrix();
		SGMatrix<float64_t> L_tmp = CMatrixOperations::get_choleksy(
			W_tmp, sW_tmp, m_ktrtr, std::exp(m_log_scale * 2.0));
		Map<MatrixXd> eigen_L(L_tmp.matrix, L_tmp.num_rows, L_tmp.num_cols);

		lik->set_dual_parameters(W_tmp, m_labels);

		//construct alpha
		SGVector<float64_t> alpha_tmp=lik->get_mu_dual_parameter();
		Map<VectorXd> eigen_alpha(alpha_tmp.vector, alpha_tmp.vlen);
		eigen_alpha=-eigen_alpha;
		//construct mu
		SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
		Map<VectorXd> eigen_mean(mean.vector, mean.vlen);
		SGVector<float64_t> mu_tmp(len);
		Map<VectorXd> eigen_mu(mu_tmp.vector, mu_tmp.vlen);
		//mu=K*alpha+m
		eigen_mu =
			eigen_K * std::exp(m_log_scale * 2.0) * eigen_alpha + eigen_mean;
		//construct s2
		MatrixXd eigen_V = eigen_L.triangularView<Upper>().adjoint().solve(
			eigen_sW.asDiagonal() * eigen_K * std::exp(m_log_scale * 2.0));
		SGVector<float64_t> s2_tmp(len);
		Map<VectorXd> eigen_s2(s2_tmp.vector, s2_tmp.vlen);
		eigen_s2 = (eigen_K.diagonal().array() * std::exp(m_log_scale * 2.0) -
			        (eigen_V.array().pow(2).colwise().sum().transpose()))
			           .abs()
			           .matrix();

		lik->set_variational_distribution(mu_tmp, s2_tmp, m_labels);

		nlml_def=get_nlml_wrapper(alpha_tmp, mu_tmp, L_tmp);

		if (nlml_new<=nlml_def)
		{
			lik->set_dual_parameters(m_W, m_labels);
			lik->set_variational_distribution(m_mu, m_s2, m_labels);
		}
	}

	if (m_alpha.vlen != m_labels->get_num_labels() || nlml_def<nlml_new)
	{
		if(m_alpha.vlen != m_labels->get_num_labels())
			m_alpha = SGVector<float64_t>(m_labels->get_num_labels());

		index_t len=m_alpha.vlen;

		m_W=SGVector<float64_t>(len);
		for (index_t i=0; i<m_W.vlen; i++)
			m_W[i]=0.5;

		lik->set_dual_parameters(m_W, m_labels);
		m_sW=SGVector<float64_t>(len);
		m_mu=SGVector<float64_t>(len);
		m_s2=SGVector<float64_t>(len);
		m_Sigma=SGMatrix<float64_t>(len, len);
		m_Sigma.zero();
		m_V=SGMatrix<float64_t>(len, len);
	}

	nlml_new=optimization();
	lik->set_variational_distribution(m_mu, m_s2, m_labels);
	TParameter* s2_param=lik->m_parameters->get_parameter("sigma2");
	m_dv=lik->get_variational_first_derivative(s2_param);
	TParameter* mu_param=lik->m_parameters->get_parameter("mu");
	m_df=lik->get_variational_first_derivative(mu_param);
}

float64_t KLDualInferenceMethod::optimization()
{
	auto minimizer=m_minimizer->as<KLDualInferenceMethodMinimizer>();
	REQUIRE(minimizer,"The minimizer must be an instance of KLDualInferenceMethodMinimizer\n");
    auto cost_fun=std::make_shared<KLDualInferenceMethodCostFunction>();
	cost_fun->set_target(shared_from_this()->as<KLDualInferenceMethod>());
	bool cleanup=false;

	minimizer->set_cost_function(cost_fun);
	float64_t nlml_opt = minimizer->minimize();
	minimizer->unset_cost_function(false);
	//cost_fun->unset_target(cleanup);

	return nlml_opt;
}

SGVector<float64_t> KLDualInferenceMethod::get_diagonal_vector()
{
	if (parameter_hash_changed())
		update();

	return SGVector<float64_t>(m_sW);
}

void KLDualInferenceMethod::update_deriv()
{
	/* get_derivative_related_cov(MatrixXd eigen_dK) does the similar job
	 * Therefore, this function body is empty
	 */
}

void KLDualInferenceMethod::update_chol()
{
	/* L is automatically updated when update_alpha is called
	 * Therefore, this function body is empty
	 */
}

void KLDualInferenceMethod::update_approx_cov()
{
	m_Sigma = CMatrixOperations::get_inverse(
		m_L, m_ktrtr, m_sW, m_V, std::exp(m_log_scale));
}

} /* namespace shogun */

