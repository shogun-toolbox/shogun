/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wu Lin, Heiko Strathmann, Fernando Iglesias, Viktor Gal
 */

#include <shogun/machine/gp/SingleLaplaceInferenceMethod.h>


#include <shogun/machine/gp/StudentsTLikelihood.h>
#include <shogun/mathematics/Math.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/lib/external/brent.h>
#endif //USE_GPL_SHOGUN
#include <shogun/mathematics/eigen3.h>
#include <shogun/optimization/FirstOrderMinimizer.h>

using namespace shogun;
using namespace Eigen;

namespace shogun
{

#ifdef USE_GPL_SHOGUN
#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** Wrapper class used for the Brent minimizer */
class PsiLine : public func_base
{
public:
	float64_t log_scale;
	MatrixXd K;
	VectorXd dalpha;
	VectorXd start_alpha;
	Map<VectorXd>* alpha;
	SGVector<float64_t>* dlp;
	SGVector<float64_t>* W;
	SGVector<float64_t>* f;
	SGVector<float64_t>* m;
	std::shared_ptr<LikelihoodModel> lik;
	std::shared_ptr<Labels> lab;

	virtual double operator() (double x)
	{
		Map<VectorXd> eigen_f(f->vector, f->vlen);
		Map<VectorXd> eigen_m(m->vector, m->vlen);

		// compute alpha=alpha+x*dalpha and f=K*alpha+m
		(*alpha)=start_alpha+x*dalpha;
		eigen_f = K * (*alpha) * std::exp(log_scale * 2.0) + eigen_m;

		// get first and second derivatives of log likelihood
		(*dlp)=lik->get_log_probability_derivative_f(lab, (*f), 1);

		(*W)=lik->get_log_probability_derivative_f(lab, (*f), 2);
		W->scale(-1.0);

		// compute psi=alpha'*(f-m)/2-lp
		float64_t result = (*alpha).dot(eigen_f-eigen_m)/2.0-
			SGVector<float64_t>::sum(lik->get_log_probability_f(lab, *f));

		return result;
	}
};
#endif //USE_GPL_SHOGUN

class SingleLaplaceInferenceMethodCostFunction: public FirstOrderCostFunction
{
public:
	SingleLaplaceInferenceMethodCostFunction():FirstOrderCostFunction() {  init(); }
	virtual ~SingleLaplaceInferenceMethodCostFunction() {  }
	void set_target(std::shared_ptr<SingleLaplaceInferenceMethod >obj)
	{
		require(obj, "Obj must set");
		if(m_obj != obj)
		{


			m_obj=obj;
		}
	}
	virtual float64_t get_cost()
	{
		require(m_obj,"Object not set");
		return m_obj->get_psi_wrt_alpha();
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
		m_obj->get_gradient_wrt_alpha(m_derivatives);
		return m_derivatives;
	}
	virtual const char* get_name() const { return "SingleLaplaceInferenceMethodCostFunction"; }
private:
	void init()
	{
		m_obj=NULL;
		m_derivatives = SGVector<float64_t>();
		SG_ADD(&m_derivatives, "SingleLaplaceInferenceMethodCostFunction__m_derivatives",
			"derivatives in SingleLaplaceInferenceMethodCostFunction");
		SG_ADD((std::shared_ptr<SGObject>*)&m_obj, "SingleLaplaceInferenceMethodCostFunction__m_obj",
			"obj in SingleLaplaceInferenceMethodCostFunction");

	}

	SGVector<float64_t> m_derivatives;
	std::shared_ptr<SingleLaplaceInferenceMethod >m_obj;
};
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


void SingleLaplaceNewtonOptimizer::set_target(std::shared_ptr<SingleLaplaceInferenceMethod >obj)
{
	require(obj, "Obj must set");
	if(m_obj != obj)
	{


		m_obj=obj;
	}
}

void SingleLaplaceNewtonOptimizer::unset_target(bool is_unref)
{
	if(is_unref)
	{

	}
	m_obj=NULL;

}

void SingleLaplaceNewtonOptimizer::init()
{
	m_obj=NULL;
	m_iter=20;
	m_tolerance=1e-6;
	m_opt_tolerance=1e-6;
	m_opt_max=10;

	SG_ADD((std::shared_ptr<SGObject>*)&m_obj, "CSingleLaplaceNewtonOptimizer__m_obj",
		"obj in CSingleLaplaceNewtonOptimizer");
	SG_ADD(&m_iter, "CSingleLaplaceNewtonOptimizer__m_iter",
		"iter in CSingleLaplaceNewtonOptimizer");
	SG_ADD(&m_tolerance, "CSingleLaplaceNewtonOptimizer__m_tolerance",
		"tolerance in CSingleLaplaceNewtonOptimizer");
	SG_ADD(&m_opt_tolerance, "CSingleLaplaceNewtonOptimizer__m_opt_tolerance",
		"opt_tolerance in CSingleLaplaceNewtonOptimizer");
	SG_ADD(&m_opt_max, "CSingleLaplaceNewtonOptimizer__m_opt_max",
		"opt_max in CSingleLaplaceNewtonOptimizer");
}

float64_t SingleLaplaceNewtonOptimizer::minimize()
{
	require(m_obj,"Object not set");
	float64_t Psi_Old=Math::INFTY;
	float64_t Psi_New=m_obj->m_Psi;

	// get mean vector and create eigen representation of it
	Map<VectorXd> eigen_mean( (m_obj->m_mean_f).vector, (m_obj->m_mean_f).vlen);

	// create eigen representation of kernel matrix
	Map<MatrixXd> eigen_ktrtr( (m_obj->m_ktrtr).matrix, (m_obj->m_ktrtr).num_rows, (m_obj->m_ktrtr).num_cols);

	Map<VectorXd> eigen_mu(m_obj->m_mu, (m_obj->m_mu).vlen);

	// compute W = -d2lp
	m_obj->m_W=m_obj->m_model->get_log_probability_derivative_f(m_obj->m_labels, m_obj->m_mu, 2);
	m_obj->m_W.scale(-1.0);

	Map<VectorXd> eigen_alpha(m_obj->m_alpha.vector, m_obj->m_alpha.vlen);

	// get first derivative of log probability function
	m_obj->m_dlp=m_obj->m_model->get_log_probability_derivative_f(m_obj->m_labels, m_obj->m_mu, 1);

	// create shogun and eigen representation of sW
	m_obj->m_sW=SGVector<float64_t>((m_obj->m_W).vlen);
	Map<VectorXd> eigen_sW((m_obj->m_sW).vector, (m_obj->m_sW).vlen);

	index_t iter=0;

	while (Psi_Old-Psi_New>m_tolerance && iter<m_iter)
	{
		Map<VectorXd> eigen_W( (m_obj->m_W).vector, (m_obj->m_W).vlen);
		Map<VectorXd> eigen_dlp( (m_obj->m_dlp).vector, (m_obj->m_dlp).vlen);

		Psi_Old = Psi_New;
		iter++;

		if (eigen_W.minCoeff() < 0)
		{
			// Suggested by Vanhatalo et. al.,
			// Gaussian Process Regression with Student's t likelihood, NIPS 2009
			// Quoted from infLaplace.m
			float64_t df;

			if (m_obj->m_model->get_model_type()==LT_STUDENTST)
			{
				auto lik = m_obj->m_model->as<StudentsTLikelihood>();
				df=lik->get_degrees_freedom();
			}
			else
				df=1;

			eigen_W+=(2.0/df)*eigen_dlp.cwiseProduct(eigen_dlp);
		}

		// compute sW = sqrt(W)
		eigen_sW=eigen_W.cwiseSqrt();

		LLT<MatrixXd> L(
			(eigen_sW * eigen_sW.transpose())
			    .cwiseProduct(
			        eigen_ktrtr * std::exp((m_obj->m_log_scale) * 2.0)) +
			MatrixXd::Identity(
			    (m_obj->m_ktrtr).num_rows, (m_obj->m_ktrtr).num_cols));

		VectorXd b=eigen_W.cwiseProduct(eigen_mu - eigen_mean)+eigen_dlp;

		VectorXd dalpha = b -
			              eigen_sW.cwiseProduct(
			                  L.solve(
			                      eigen_sW.cwiseProduct(
			                          eigen_ktrtr * b *
			                          std::exp((m_obj->m_log_scale) * 2.0)))) -
			              eigen_alpha;

#ifdef USE_GPL_SHOGUN
		// perform Brent's optimization
		PsiLine func;

		func.log_scale=m_obj->m_log_scale;
		func.K=eigen_ktrtr;
		func.dalpha=dalpha;
		func.start_alpha=eigen_alpha;
		func.alpha=&eigen_alpha;
		func.dlp=&(m_obj->m_dlp);
		func.f=&(m_obj->m_mu);
		func.m=&(m_obj->m_mean_f);
		func.W=&(m_obj->m_W);
		func.lik=m_obj->m_model;
		func.lab=m_obj->m_labels;

		float64_t x;
		Psi_New=local_min(0, m_opt_max, m_opt_tolerance, func, x);
#else
		gpl_only(SOURCE_LOCATION);
#endif //USE_GPL_SHOGUN
	}

	if (Psi_Old-Psi_New>m_tolerance && iter>=m_iter)
	{
		io::warn("Max iterations ({}) reached, but convergence level ({}) is not yet below tolerance ({})", m_iter, Psi_Old-Psi_New, m_tolerance);
	}
	return Psi_New;
}

SingleLaplaceInferenceMethod::SingleLaplaceInferenceMethod() : LaplaceInference()
{
	init();
}

SingleLaplaceInferenceMethod::SingleLaplaceInferenceMethod(std::shared_ptr<Kernel> kern,
		std::shared_ptr<Features> feat, std::shared_ptr<MeanFunction> m, std::shared_ptr<Labels> lab, std::shared_ptr<LikelihoodModel> mod)
		: LaplaceInference(kern, feat, m, lab, mod)
{
	init();
}

void SingleLaplaceInferenceMethod::init()
{
	m_Psi=0;
	SG_ADD(&m_Psi, "Psi", "posterior log likelihood without constant terms");
	SG_ADD(&m_sW, "sW", "square root of W");
	SG_ADD(&m_d2lp, "d2lp", "second derivative of log likelihood with respect to function location");
	SG_ADD(&m_d3lp, "d3lp", "third derivative of log likelihood with respect to function location");
	register_minimizer(std::make_shared<SingleLaplaceNewtonOptimizer>());
}

SGVector<float64_t> SingleLaplaceInferenceMethod::get_diagonal_vector()
{
	if (parameter_hash_changed())
		update();

	return SGVector<float64_t>(m_sW);
}

std::shared_ptr<SingleLaplaceInferenceMethod> SingleLaplaceInferenceMethod::obtain_from_generic(
		std::shared_ptr<Inference> inference)
{
	if (inference==NULL)
		return NULL;

	if (inference->get_inference_type()!=INF_LAPLACE_SINGLE)
		error("Provided inference is not of type SingleLaplaceInferenceMethod");

	return inference->as<SingleLaplaceInferenceMethod>();
}

SingleLaplaceInferenceMethod::~SingleLaplaceInferenceMethod()
{
}

float64_t SingleLaplaceInferenceMethod::get_negative_log_marginal_likelihood()
{
	if (parameter_hash_changed())
		update();

	// create eigen representations alpha, f, W, L
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	Map<VectorXd> eigen_W(m_W.vector, m_W.vlen);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	// get mean vector and create eigen representation of it
	SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);

	// get log likelihood
	float64_t lp=SGVector<float64_t>::sum(m_model->get_log_probability_f(m_labels,
		m_mu));

	float64_t result;

	if (eigen_W.minCoeff()<0)
	{
		Map<VectorXd> eigen_sW(m_sW.vector, m_sW.vlen);
		Map<MatrixXd> eigen_ktrtr(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

		FullPivLU<MatrixXd> lu(
			MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols) +
			eigen_ktrtr * std::exp(m_log_scale * 2.0) * eigen_sW.asDiagonal());

		result=(eigen_alpha.dot(eigen_mu-eigen_mean))/2.0-
			lp+log(lu.determinant())/2.0;
	}
	else
	{
		result=eigen_alpha.dot(eigen_mu-eigen_mean)/2.0-lp+
			eigen_L.diagonal().array().log().sum();
	}

	return result;
}

void SingleLaplaceInferenceMethod::update_approx_cov()
{
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> eigen_sW(m_sW.vector, m_sW.vlen);

	m_Sigma=SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_Sigma(m_Sigma.matrix, m_Sigma.num_rows,	m_Sigma.num_cols);

	// compute V = L^(-1) * W^(1/2) * K, using upper triangular factor L^T
	MatrixXd eigen_V = eigen_L.triangularView<Upper>().adjoint().solve(
		eigen_sW.asDiagonal() * eigen_K * std::exp(m_log_scale * 2.0));

	// compute covariance matrix of the posterior:
	// Sigma = K - K * W^(1/2) * (L * L^T)^(-1) * W^(1/2) * K =
	// K - (K * W^(1/2)) * (L^T)^(-1) * L^(-1) * W^(1/2) * K =
	// K - (W^(1/2) * K)^T * (L^(-1))^T * L^(-1) * W^(1/2) * K = K - V^T * V
	eigen_Sigma =
		eigen_K * std::exp(m_log_scale * 2.0) - eigen_V.adjoint() * eigen_V;
}

void SingleLaplaceInferenceMethod::update_chol()
{
	// get log probability derivatives
	m_dlp=m_model->get_log_probability_derivative_f(m_labels, m_mu, 1);
	m_d2lp=m_model->get_log_probability_derivative_f(m_labels, m_mu, 2);
	m_d3lp=m_model->get_log_probability_derivative_f(m_labels, m_mu, 3);

	// W = -d2lp
	m_W=m_d2lp.clone();
	m_W.scale(-1.0);
	m_sW=SGVector<float64_t>(m_W.vlen);

	// compute sW
	Map<VectorXd> eigen_W(m_W.vector, m_W.vlen);
	Map<VectorXd> eigen_sW(m_sW.vector, m_sW.vlen);

	if (eigen_W.minCoeff()>0)
		eigen_sW=eigen_W.cwiseSqrt();
	else
		//post.sW = sqrt(abs(W)).*sign(W);
		eigen_sW=((eigen_W.array().abs()+eigen_W.array())/2).sqrt()-((eigen_W.array().abs()-eigen_W.array())/2).sqrt();

	// create eigen representation of kernel matrix
	Map<MatrixXd> eigen_ktrtr(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	// create shogun and eigen representation of posterior cholesky
	m_L=SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	if (eigen_W.minCoeff() < 0)
	{
		//A = eye(n)+K.*repmat(w',n,1);
		FullPivLU<MatrixXd> lu(
			MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols) +
			eigen_ktrtr * std::exp(m_log_scale * 2.0) * eigen_W.asDiagonal());
		// compute cholesky: L = -(K + 1/W)^-1
		//-iA = -inv(A)
		eigen_L=-lu.inverse();
		// -repmat(w,1,n).*iA == (-iA'.*repmat(w',n,1))'
		eigen_L=eigen_W.asDiagonal()*eigen_L;
	}
	else
	{
		// compute cholesky: L = chol(sW * sW' .* K + I)
		LLT<MatrixXd> L(
			(eigen_sW * eigen_sW.transpose())
			    .cwiseProduct(eigen_ktrtr * std::exp(m_log_scale * 2.0)) +
			MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));

		eigen_L = L.matrixU();
	}
}

void SingleLaplaceInferenceMethod::update()
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


void SingleLaplaceInferenceMethod::update_init()
{
	float64_t Psi_New;
	float64_t Psi_Def;
	// get mean vector and create eigen representation of it
	m_mean_f = m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mean(m_mean_f.vector, m_mean_f.vlen);

	// create eigen representation of kernel matrix
	Map<MatrixXd> eigen_ktrtr(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	// create shogun and eigen representation of function vector
	m_mu=SGVector<float64_t>(m_mean_f.vlen);
	Map<VectorXd> eigen_mu(m_mu, m_mu.vlen);

	if (m_alpha.vlen!=m_labels->get_num_labels())
	{
		// set alpha a zero vector
		m_alpha=SGVector<float64_t>(m_labels->get_num_labels());
		m_alpha.zero();

		// f = mean, if length of alpha and length of y doesn't match
		eigen_mu=eigen_mean;

		Psi_New=-SGVector<float64_t>::sum(m_model->get_log_probability_f(
			m_labels, m_mu));
	}
	else
	{
		Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

		// compute f = K * alpha + m
		eigen_mu = eigen_ktrtr * std::exp(m_log_scale * 2.0) * eigen_alpha +
			       eigen_mean;

		Psi_New=eigen_alpha.dot(eigen_mu-eigen_mean)/2.0-
			SGVector<float64_t>::sum(m_model->get_log_probability_f(m_labels, m_mu));

		Psi_Def=-SGVector<float64_t>::sum(m_model->get_log_probability_f(m_labels, m_mean_f));

		// if default is better, then use it
		if (Psi_Def < Psi_New)
		{
			m_alpha.zero();
			eigen_mu=eigen_mean;
			Psi_New=Psi_Def;
		}
	}
	m_Psi=Psi_New;
}


void SingleLaplaceInferenceMethod::register_minimizer(std::shared_ptr<Minimizer> minimizer)
{
	require(minimizer, "Minimizer must set");
	if (!std::dynamic_pointer_cast<SingleLaplaceNewtonOptimizer>(minimizer))
	{
		auto opt= std::dynamic_pointer_cast<FirstOrderMinimizer>(minimizer);
		require(opt, "The provided minimizer is not supported");
	}
	Inference::register_minimizer(minimizer);
}

void SingleLaplaceInferenceMethod::update_alpha()
{
	auto opt=std::dynamic_pointer_cast<SingleLaplaceNewtonOptimizer>(m_minimizer);
	bool cleanup=false;
	if (opt)
	{
		opt->set_target(shared_from_this()->as<SingleLaplaceInferenceMethod>());
		opt->minimize();
	}
	else
	{
		auto minimizer= std::dynamic_pointer_cast<FirstOrderMinimizer>(m_minimizer);
		require(minimizer, "The provided minimizer is not supported");
#ifdef USE_GPL_SHOGUN
		auto cost_fun=std::make_shared<SingleLaplaceInferenceMethodCostFunction>();
		cost_fun->set_target(shared_from_this()->as<SingleLaplaceInferenceMethod>());
		minimizer->set_cost_function(cost_fun);
		minimizer->minimize();
		minimizer->unset_cost_function(false);

#else
		gpl_only(SOURCE_LOCATION);
#endif //USE_GPL_SHOGUN
	}
	// get mean vector and create eigen representation of it
	Map<VectorXd> eigen_mean(m_mean_f.vector, m_mean_f.vlen);

	// create eigen representation of kernel matrix
	Map<MatrixXd> eigen_ktrtr(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	Map<VectorXd> eigen_mu(m_mu, m_mu.vlen);

	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	// compute f = K * alpha + m
	eigen_mu =
		eigen_ktrtr * std::exp(m_log_scale * 2.0) * eigen_alpha + eigen_mean;
}

void SingleLaplaceInferenceMethod::update_deriv()
{
	// create eigen representation of W, sW, dlp, d3lp, K, alpha and L
	Map<VectorXd> eigen_W(m_W.vector, m_W.vlen);
	Map<VectorXd> eigen_sW(m_sW.vector, m_sW.vlen);
	Map<VectorXd> eigen_dlp(m_dlp.vector, m_dlp.vlen);
	Map<VectorXd> eigen_d3lp(m_d3lp.vector, m_d3lp.vlen);
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	// create shogun and eigen representation of matrix Z
	m_Z=SGMatrix<float64_t>(m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_Z(m_Z.matrix, m_Z.num_rows, m_Z.num_cols);

	// create shogun and eigen representation of the vector g
	m_g=SGVector<float64_t>(m_Z.num_rows);
	Map<VectorXd> eigen_g(m_g.vector, m_g.vlen);

	if (eigen_W.minCoeff()<0)
	{
		eigen_Z=-eigen_L;

		// compute iA = (I + K * diag(W))^-1
		FullPivLU<MatrixXd> lu(
			MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols) +
			eigen_K * std::exp(m_log_scale * 2.0) * eigen_W.asDiagonal());
		MatrixXd iA=lu.inverse();

		// compute derivative ln|L'*L| wrt W: g=sum(iA.*K,2)/2
		eigen_g = (iA.cwiseProduct(eigen_K * std::exp(m_log_scale * 2.0)))
			          .rowwise()
			          .sum() /
			      2.0;
	}
	else
	{
		// solve L'*L*Z=diag(sW) and compute Z=diag(sW)*Z
		eigen_Z=eigen_L.triangularView<Upper>().adjoint().solve(
				MatrixXd(eigen_sW.asDiagonal()));
		eigen_Z=eigen_L.triangularView<Upper>().solve(eigen_Z);
		eigen_Z=eigen_sW.asDiagonal()*eigen_Z;

		// solve L'*C=diag(sW)*K
		MatrixXd C = eigen_L.triangularView<Upper>().adjoint().solve(
			eigen_sW.asDiagonal() * eigen_K * std::exp(m_log_scale * 2.0));

		// compute derivative ln|L'*L| wrt W: g=(diag(K)-sum(C.^2,1)')/2
		eigen_g = (eigen_K.diagonal() * std::exp(m_log_scale * 2.0) -
			       (C.cwiseProduct(C)).colwise().sum().adjoint()) /
			      2.0;
	}

	// create shogun and eigen representation of the vector dfhat
	m_dfhat=SGVector<float64_t>(m_g.vlen);
	Map<VectorXd> eigen_dfhat(m_dfhat.vector, m_dfhat.vlen);

	// compute derivative of nlZ wrt fhat
	eigen_dfhat=eigen_g.cwiseProduct(eigen_d3lp);
}

SGVector<float64_t> SingleLaplaceInferenceMethod::get_derivative_wrt_inference_method(
		const TParameter* param)
{
	require(!strcmp(param->m_name, "log_scale"), "Can't compute derivative of "
			"the nagative log marginal likelihood wrt {}.{} parameter",
			get_name(), param->m_name);

	// create eigen representation of K, Z, dfhat, dlp and alpha
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_Z(m_Z.matrix, m_Z.num_rows, m_Z.num_cols);
	Map<VectorXd> eigen_dfhat(m_dfhat.vector, m_dfhat.vlen);
	Map<VectorXd> eigen_dlp(m_dlp.vector, m_dlp.vlen);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	SGVector<float64_t> result(1);

	// compute derivative K wrt scale
	// compute dnlZ=sum(sum(Z.*dK))/2-alpha'*dK*alpha/2
	result[0]=(eigen_Z.cwiseProduct(eigen_K)).sum()/2.0-
		(eigen_alpha.adjoint()*eigen_K).dot(eigen_alpha)/2.0;

	// compute b=dK*dlp
	VectorXd b=eigen_K*eigen_dlp;

	// compute dnlZ=dnlZ-dfhat'*(b-K*(Z*b))
	result[0] = result[0] -
		        eigen_dfhat.dot(
		            b - eigen_K * std::exp(m_log_scale * 2.0) * (eigen_Z * b));
	result[0] *= std::exp(m_log_scale * 2.0) * 2.0;

	return result;
}

SGVector<float64_t> SingleLaplaceInferenceMethod::get_derivative_wrt_likelihood_model(
		const TParameter* param)
{
	// create eigen representation of K, Z, g and dfhat
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_Z(m_Z.matrix, m_Z.num_rows, m_Z.num_cols);
	Map<VectorXd> eigen_g(m_g.vector, m_g.vlen);
	Map<VectorXd> eigen_dfhat(m_dfhat.vector, m_dfhat.vlen);

	// get derivatives wrt likelihood model parameters
	SGVector<float64_t> lp_dhyp=m_model->get_first_derivative(m_labels,
			m_mu, param);
	SGVector<float64_t> dlp_dhyp=m_model->get_second_derivative(m_labels,
			m_mu, param);
	SGVector<float64_t> d2lp_dhyp=m_model->get_third_derivative(m_labels,
			m_mu, param);

	// create eigen representation of the derivatives
	Map<VectorXd> eigen_lp_dhyp(lp_dhyp.vector, lp_dhyp.vlen);
	Map<VectorXd> eigen_dlp_dhyp(dlp_dhyp.vector, dlp_dhyp.vlen);
	Map<VectorXd> eigen_d2lp_dhyp(d2lp_dhyp.vector, d2lp_dhyp.vlen);

	SGVector<float64_t> result(1);

	// compute b vector
	VectorXd b=eigen_K*eigen_dlp_dhyp;

	// compute dnlZ=-g'*d2lp_dhyp-sum(lp_dhyp)-dfhat'*(b-K*(Z*b))
	result[0] = -eigen_g.dot(eigen_d2lp_dhyp) - eigen_lp_dhyp.sum() -
		        eigen_dfhat.dot(
		            b - eigen_K * std::exp(m_log_scale * 2.0) * (eigen_Z * b));

	return result;
}

SGVector<float64_t> SingleLaplaceInferenceMethod::get_derivative_wrt_kernel(
		const TParameter* param)
{
	// create eigen representation of K, Z, dfhat, dlp and alpha
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_Z(m_Z.matrix, m_Z.num_rows, m_Z.num_cols);
	Map<VectorXd> eigen_dfhat(m_dfhat.vector, m_dfhat.vlen);
	Map<VectorXd> eigen_dlp(m_dlp.vector, m_dlp.vlen);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	require(param, "Param not set");
	SGVector<float64_t> result;
	int64_t len=const_cast<TParameter *>(param)->m_datatype.get_num_elements();
	result=SGVector<float64_t>(len);

	for (index_t i=0; i<result.vlen; i++)
	{
		SGMatrix<float64_t> dK;

		if (result.vlen==1)
			dK=m_kernel->get_parameter_gradient(param);
		else
			dK=m_kernel->get_parameter_gradient(param, i);

		Map<MatrixXd> eigen_dK(dK.matrix, dK.num_rows, dK.num_cols);

		// compute dnlZ=sum(sum(Z.*dK))/2-alpha'*dK*alpha/2
		result[i]=(eigen_Z.cwiseProduct(eigen_dK)).sum()/2.0-
			(eigen_alpha.adjoint()*eigen_dK).dot(eigen_alpha)/2.0;

		// compute b=dK*dlp
		VectorXd b=eigen_dK*eigen_dlp;

		// compute dnlZ=dnlZ-dfhat'*(b-K*(Z*b))
		result[i] =
			result[i] -
			eigen_dfhat.dot(
			    b - eigen_K * std::exp(m_log_scale * 2.0) * (eigen_Z * b));
		result[i] *= std::exp(m_log_scale * 2.0);
	}

	return result;
}

SGVector<float64_t> SingleLaplaceInferenceMethod::get_derivative_wrt_mean(
		const TParameter* param)
{
	// create eigen representation of K, Z, dfhat and alpha
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_Z(m_Z.matrix, m_Z.num_rows, m_Z.num_cols);
	Map<VectorXd> eigen_dfhat(m_dfhat.vector, m_dfhat.vlen);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	require(param, "Param not set");
	SGVector<float64_t> result;
	int64_t len=const_cast<TParameter *>(param)->m_datatype.get_num_elements();
	result=SGVector<float64_t>(len);

	for (index_t i=0; i<result.vlen; i++)
	{
		SGVector<float64_t> dmu;

		if (result.vlen==1)
			dmu=m_mean->get_parameter_derivative(m_features, param);
		else
			dmu=m_mean->get_parameter_derivative(m_features, param, i);

		Map<VectorXd> eigen_dmu(dmu.vector, dmu.vlen);

		// compute dnlZ=-alpha'*dm-dfhat'*(dm-K*(Z*dm))
		result[i] =
			-eigen_alpha.dot(eigen_dmu) -
			eigen_dfhat.dot(
			    eigen_dmu -
			    eigen_K * std::exp(m_log_scale * 2.0) * (eigen_Z * eigen_dmu));
	}

	return result;
}

SGVector<float64_t> SingleLaplaceInferenceMethod::get_posterior_mean()
{
	compute_gradient();

	SGVector<float64_t> res(m_mu.vlen);
	Map<VectorXd> eigen_res(res.vector, res.vlen);

	Map<VectorXd> eigen_mu(m_mu, m_mu.vlen);
	SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);
	eigen_res=eigen_mu-eigen_mean;

	return res;
}


float64_t SingleLaplaceInferenceMethod::get_psi_wrt_alpha()
{
	Eigen::Map<Eigen::VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	SGVector<float64_t> f(m_alpha.vlen);
	Eigen::Map<Eigen::VectorXd> eigen_f(f.vector, f.vlen);
	Eigen::Map<Eigen::MatrixXd> kernel(m_ktrtr.matrix,
		m_ktrtr.num_rows,
		m_ktrtr.num_cols);
	Eigen::Map<Eigen::VectorXd> eigen_mean_f(m_mean_f.vector,
		m_mean_f.vlen);
	/* f = K * alpha + mean_f given alpha*/
	eigen_f =
		kernel * ((eigen_alpha)*std::exp(m_log_scale * 2.0)) + eigen_mean_f;

	/* psi = 0.5 * alpha .* (f - m) - sum(dlp)*/
	float64_t psi = eigen_alpha.dot(eigen_f - eigen_mean_f) * 0.5;
	psi -= SGVector<float64_t>::sum(m_model->get_log_probability_f(m_labels, f));
	return psi;
}

void SingleLaplaceInferenceMethod::get_gradient_wrt_alpha(SGVector<float64_t> gradient)
{
	require(gradient.vlen==m_alpha.vlen,
		"The length of gradients ({}) should the same as the length of parameters ({})",
		gradient.vlen, m_alpha.vlen);

	Eigen::Map<Eigen::VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	Eigen::Map<Eigen::VectorXd> eigen_gradient(gradient.vector, gradient.vlen);
	SGVector<float64_t> f(m_alpha.vlen);
	Eigen::Map<Eigen::VectorXd> eigen_f(f.vector, f.vlen);
	Eigen::Map<Eigen::MatrixXd> kernel(m_ktrtr.matrix,
		m_ktrtr.num_rows,
		m_ktrtr.num_cols);
	Eigen::Map<Eigen::VectorXd> eigen_mean_f(m_mean_f.vector,
		m_mean_f.vlen);

	/* f = K * alpha + mean_f given alpha*/
	eigen_f =
		kernel * ((eigen_alpha)*std::exp(m_log_scale * 2.0)) + eigen_mean_f;

	SGVector<float64_t> dlp_f =
		m_model->get_log_probability_derivative_f(m_labels, f, 1);

	Eigen::Map<Eigen::VectorXd> eigen_dlp_f(dlp_f.vector, dlp_f.vlen);

	/* g_alpha = K * (alpha - dlp_f)*/
	eigen_gradient =
		kernel * ((eigen_alpha - eigen_dlp_f) * std::exp(m_log_scale * 2.0));
}


}

