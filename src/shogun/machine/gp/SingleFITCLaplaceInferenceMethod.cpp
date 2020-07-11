/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Wu Lin
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
 */

#include <shogun/machine/gp/SingleFITCLaplaceInferenceMethod.h>

#include <shogun/machine/gp/StudentsTLikelihood.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/visitors/ShapeVisitor.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/lib/external/brent.h>
#endif //USE_GPL_SHOGUN
#include <shogun/mathematics/eigen3.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/optimization/FirstOrderMinimizer.h>

#include <utility>

using namespace shogun;
using namespace Eigen;

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#ifdef USE_GPL_SHOGUN
/** Wrapper class used for the Brent minimizer */
class CFITCPsiLine : public func_base
{
public:
	float64_t log_scale;
	VectorXd dalpha;
	VectorXd start_alpha;
	SGVector<float64_t>* alpha;
	SGVector<float64_t>* dlp;
	SGVector<float64_t>* W;
	SGVector<float64_t>* f;
	SGVector<float64_t>* m;
	std::shared_ptr<LikelihoodModel> lik;
	std::shared_ptr<Labels> lab;
	std::shared_ptr<SingleFITCLaplaceInferenceMethod >inf;

	double operator() (double x) override
	{
		//time complexity O(m*n)
		Map<VectorXd> eigen_f(f->vector, f->vlen);
		Map<VectorXd> eigen_m(m->vector, m->vlen);
		Map<VectorXd> eigen_alpha(alpha->vector, alpha->vlen);

		//alpha = alpha + s*dalpha;
		eigen_alpha=start_alpha+x*dalpha;
		SGVector<float64_t> tmp=inf->compute_mvmK(*alpha);
		Map<VectorXd> eigen_tmp(tmp.vector, tmp.vlen);
		//f = mvmK(alpha,V,d0)+m;
		eigen_f=eigen_tmp+eigen_m;

		// get first and second derivatives of log likelihood
		(*dlp)=lik->get_log_probability_derivative_f(lab, (*f), 1);

		(*W)=lik->get_log_probability_derivative_f(lab, (*f), 2);
		W->scale(-1.0);

		// compute psi=alpha'*(f-m)/2-lp
		float64_t result = eigen_alpha.dot(eigen_f-eigen_m)/2.0-
			SGVector<float64_t>::sum(lik->get_log_probability_f(lab, *f));

		return result;
	}
};
#endif //USE_GPL_SHOGUN

class SingleFITCLaplaceInferenceMethodCostFunction: public FirstOrderCostFunction
{
public:
	SingleFITCLaplaceInferenceMethodCostFunction():FirstOrderCostFunction() {  init(); }
	~SingleFITCLaplaceInferenceMethodCostFunction() override { clean(); }
	void set_target(const std::shared_ptr<SingleFITCLaplaceInferenceMethod >&obj)
	{
		require(obj, "Obj must set");
		if(m_obj != obj)
		{


			m_obj=obj;
		}
	}

	void clean()
	{

	}

	float64_t get_cost() override
	{
		require(m_obj,"Object not set");
		return m_obj->get_psi_wrt_alpha();
	}
	void unset_target(bool is_unref)
	{
		if(is_unref)
		{

		}
		m_obj=NULL;
	}
	SGVector<float64_t> obtain_variable_reference() override
	{
		require(m_obj,"Object not set");
		m_derivatives = SGVector<float64_t>((m_obj->m_al).vlen);
		return m_obj->m_al;
	}
	SGVector<float64_t> get_gradient() override
	{
		require(m_obj,"Object not set");
		m_obj->get_gradient_wrt_alpha(m_derivatives);
		return m_derivatives;
	}
	const char* get_name() const override { return "SingleFITCLaplaceInferenceMethodCostFunction"; }
private:
	void init()
	{
		m_obj=NULL;
		m_derivatives = SGVector<float64_t>();
		SG_ADD(&m_derivatives, "SingleFITCLaplaceInferenceMethodCostFunction__m_derivatives",
			"derivatives in SingleFITCLaplaceInferenceMethodCostFunction");
		SG_ADD((std::shared_ptr<SGObject>*)&m_obj, "SingleFITCLaplaceInferenceMethodCostFunction__m_obj",
			"obj in SingleFITCLaplaceInferenceMethodCostFunction");
	}

	SGVector<float64_t> m_derivatives;
	std::shared_ptr<SingleFITCLaplaceInferenceMethod >m_obj;
};

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

void SingleFITCLaplaceNewtonOptimizer::set_target(const std::shared_ptr<SingleFITCLaplaceInferenceMethod >&obj)
{
	require(obj, "Obj must set");
	if(m_obj != obj)
	{


		m_obj=obj;
	}
}

void SingleFITCLaplaceNewtonOptimizer::unset_target(bool is_unref)
{
	if(is_unref)
	{

	}
	m_obj=NULL;

}

void SingleFITCLaplaceNewtonOptimizer::init()
{
	m_obj=NULL;
	m_iter=20;
	m_tolerance=1e-6;
	m_opt_tolerance=1e-6;
	m_opt_max=10;

	SG_ADD((std::shared_ptr<SGObject>*)&m_obj, "CSingleFITCLaplaceNewtonOptimizer__m_obj",
		"obj in CSingleFITCLaplaceNewtonOptimizer");
	SG_ADD(&m_iter, "CSingleFITCLaplaceNewtonOptimizer__m_iter",
		"iter in CSingleFITCLaplaceNewtonOptimizer");
	SG_ADD(&m_tolerance, "CSingleFITCLaplaceNewtonOptimizer__m_tolerance",
		"tolerance in CSingleFITCLaplaceNewtonOptimizer");
	SG_ADD(&m_opt_tolerance, "CSingleFITCLaplaceNewtonOptimizer__m_opt_tolerance",
		"opt_tolerance in CSingleFITCLaplaceNewtonOptimizer");
	SG_ADD(&m_opt_max, "CSingleFITCLaplaceNewtonOptimizer__m_opt_max",
		"opt_max in CSingleFITCLaplaceNewtonOptimizer");
}

float64_t SingleFITCLaplaceNewtonOptimizer::minimize()
{
	require(m_obj,"Object not set");
	//time complexity O(m^2*n);
	Map<MatrixXd> eigen_kuu((m_obj->m_kuu).matrix, (m_obj->m_kuu).num_rows, (m_obj->m_kuu).num_cols);
	Map<MatrixXd> eigen_V((m_obj->m_V).matrix, (m_obj->m_V).num_rows, (m_obj->m_V).num_cols);
	Map<VectorXd> eigen_dg((m_obj->m_dg).vector, (m_obj->m_dg).vlen);
	Map<MatrixXd> eigen_R0((m_obj->m_chol_R0).matrix, (m_obj->m_chol_R0).num_rows, (m_obj->m_chol_R0).num_cols);
	Map<VectorXd> eigen_mu(m_obj->m_mu, (m_obj->m_mu).vlen);

	SGVector<float64_t> mean=m_obj->m_mean->get_mean_vector(m_obj->m_features);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);

	float64_t Psi_Old=Math::INFTY;
	float64_t Psi_New=m_obj->m_Psi;

	// compute W = -d2lp
	m_obj->m_W=m_obj->m_model->get_log_probability_derivative_f(m_obj->m_labels, m_obj->m_mu, 2);
	m_obj->m_W.scale(-1.0);

	//n-by-1 vector
	Map<VectorXd> eigen_al((m_obj->m_al).vector, (m_obj->m_al).vlen);

	// get first derivative of log probability function
	m_obj->m_dlp=m_obj->m_model->get_log_probability_derivative_f(m_obj->m_labels, m_obj->m_mu, 1);

	index_t iter=0;

	m_obj->m_Wneg=false;
	while (Psi_Old-Psi_New>m_tolerance && iter<m_iter)
	{
		//time complexity O(m^2*n)
		Map<VectorXd> eigen_W((m_obj->m_W).vector, (m_obj->m_W).vlen);
		Map<VectorXd> eigen_dlp((m_obj->m_dlp).vector, (m_obj->m_dlp).vlen);

		Psi_Old = Psi_New;
		iter++;

		if (eigen_W.minCoeff() < 0)
		{
			// Suggested by Vanhatalo et. al.,
			// Gaussian Process Regression with Student's t likelihood, NIPS 2009
			// Quoted from infFITC_Laplace.m
			float64_t df;

			if (m_obj->m_model->get_model_type()==LT_STUDENTST)
			{
				auto lik = m_obj->m_model->as<StudentsTLikelihood>();
				df=lik->get_degrees_freedom();
			}
			else
				df=1;
			eigen_W+=(2.0/(df+1))*eigen_dlp.cwiseProduct(eigen_dlp);
		}

		//b = W.*(f-m) + dlp;
		VectorXd b=eigen_W.cwiseProduct(eigen_mu-eigen_mean)+eigen_dlp;

		//dd = 1./(1+W.*d0);
		VectorXd dd=MatrixXd::Ones(b.rows(),1).cwiseQuotient(eigen_W.cwiseProduct(eigen_dg)+MatrixXd::Ones(b.rows(),1));

		VectorXd eigen_t=eigen_W.cwiseProduct(dd);
		//m-by-m matrix
		SGMatrix<float64_t> tmp( (m_obj->m_V).num_rows, (m_obj->m_V).num_rows);
		Map<MatrixXd> eigen_tmp(tmp.matrix, tmp.num_rows, tmp.num_cols);
		//eye(nu)+(V.*repmat((W.*dd)',nu,1))*V'
		eigen_tmp=eigen_V*eigen_t.asDiagonal()*eigen_V.transpose()+MatrixXd::Identity(tmp.num_rows,tmp.num_rows);
		tmp=m_obj->get_chol_inv(tmp);
		//chol_inv(eye(nu)+(V.*repmat((W.*dd)',nu,1))*V')
		Map<MatrixXd> eigen_tmp2(tmp.matrix, tmp.num_rows, tmp.num_cols);
		//RV = chol_inv(eye(nu)+(V.*repmat((W.*dd)',nu,1))*V')*V;
		// m-by-n matrix
		MatrixXd eigen_RV=eigen_tmp2*eigen_V;
		//dalpha = dd.*b - (W.*dd).*(RV'*(RV*(dd.*b))) - alpha; % Newt dir + line search
		VectorXd dalpha=dd.cwiseProduct(b)-eigen_t.cwiseProduct(eigen_RV.transpose()*(eigen_RV*(dd.cwiseProduct(b))))-eigen_al;
#ifdef USE_GPL_SHOGUN
		//perform Brent's optimization
		CFITCPsiLine func;

		func.log_scale=m_obj->m_log_scale;
		func.dalpha=dalpha;
		func.start_alpha=eigen_al;
		func.alpha=&(m_obj->m_al);
		func.dlp=&(m_obj->m_dlp);
		func.f=&(m_obj->m_mu);
		func.m=&mean;
		func.W=&(m_obj->m_W);
		func.lik=m_obj->m_model;
		func.lab=m_obj->m_labels;
		func.inf=m_obj;

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

SingleFITCLaplaceInferenceMethod::SingleFITCLaplaceInferenceMethod() : SingleFITCInference()
{
	init();
}

SingleFITCLaplaceInferenceMethod::SingleFITCLaplaceInferenceMethod(std::shared_ptr<Kernel> kern, std::shared_ptr<Features> feat,
	std::shared_ptr<MeanFunction> m, std::shared_ptr<Labels> lab, std::shared_ptr<LikelihoodModel> mod, std::shared_ptr<Features> lat)
: SingleFITCInference(std::move(kern), std::move(feat), std::move(m), std::move(lab), std::move(mod), std::move(lat))
{
	init();
}

void SingleFITCLaplaceInferenceMethod::init()
{
	m_Psi=0;
	m_Wneg=false;

	SG_ADD(&m_dlp, "dlp", "derivative of log likelihood with respect to function location");
	SG_ADD(&m_W, "W", "the noise matrix");

	SG_ADD(&m_sW, "sW", "square root of W");
	SG_ADD(&m_d2lp, "d2lp", "second derivative of log likelihood with respect to function location");
	SG_ADD(&m_d3lp, "d3lp", "third derivative of log likelihood with respect to function location");
	SG_ADD(&m_chol_R0, "chol_R0", "Cholesky of inverse covariance of inducing features");
	SG_ADD(&m_dfhat, "dfhat", "derivative of negative log (approximated) marginal likelihood wrt f");
	SG_ADD(&m_g, "g", "variable g defined in infFITC_Laplace.m");
	SG_ADD(&m_dg, "dg", "variable d0 defined in infFITC_Laplace.m");
	SG_ADD(&m_Psi, "Psi", "the negative log likelihood without constant terms used in Newton's method");
	SG_ADD(&m_Wneg, "Wneg", "whether W contains negative elements");

	register_minimizer(std::make_shared<SingleFITCLaplaceNewtonOptimizer>());
}

void SingleFITCLaplaceInferenceMethod::compute_gradient()
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

void SingleFITCLaplaceInferenceMethod::update()
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

SGVector<float64_t> SingleFITCLaplaceInferenceMethod::get_diagonal_vector()
{
	if (parameter_hash_changed())
		update();

	return SGVector<float64_t>(m_sW);
}

SingleFITCLaplaceInferenceMethod::~SingleFITCLaplaceInferenceMethod()
{
}

SGVector<float64_t> SingleFITCLaplaceInferenceMethod::compute_mvmZ(SGVector<float64_t> x)
{
	//time complexity O(m*n)
	Map<MatrixXd> eigen_Rvdd(m_Rvdd.matrix, m_Rvdd.num_rows, m_Rvdd.num_cols);
	Map<VectorXd> eigen_t(m_t.vector, m_t.vlen);
	Map<VectorXd> eigen_x(x.vector, x.vlen);

	SGVector<float64_t> res(x.vlen);
	Map<VectorXd> eigen_res(res.vector, res.vlen);

	//Zx = t.*x - RVdd'*(RVdd*x);
	eigen_res=eigen_x.cwiseProduct(eigen_t)-eigen_Rvdd.transpose()*(eigen_Rvdd*eigen_x);
	return res;
}

SGVector<float64_t> SingleFITCLaplaceInferenceMethod::compute_mvmK(SGVector<float64_t> al)
{
	//time complexity O(m*n)
	Map<MatrixXd> eigen_V(m_V.matrix, m_V.num_rows, m_V.num_cols);
	Map<VectorXd> eigen_dg(m_dg.vector, m_dg.vlen);
	Map<VectorXd> eigen_al(al.vector, al.vlen);

	SGVector<float64_t> res(al.vlen);
	Map<VectorXd> eigen_res(res.vector, res.vlen);

	//Kal = V'*(V*al) + d0.*al;
	eigen_res= eigen_V.transpose()*(eigen_V*eigen_al)+eigen_dg.cwiseProduct(eigen_al);
	return res;
}

std::shared_ptr<SingleFITCLaplaceInferenceMethod> SingleFITCLaplaceInferenceMethod::obtain_from_generic(
		const std::shared_ptr<Inference>& inference)
{
	require(inference!=NULL, "Inference should be not NULL");

	if (inference->get_inference_type()!=INF_FITC_LAPLACE_SINGLE)
		error("Provided inference is not of type SingleFITCLaplaceInferenceMethod!");
	return inference->as<SingleFITCLaplaceInferenceMethod>();
}

SGMatrix<float64_t> SingleFITCLaplaceInferenceMethod::get_chol_inv(SGMatrix<float64_t> mtx)
{
	//time complexity O(m^3), where mtx is a m-by-m matrix
	require(mtx.num_rows==mtx.num_cols, "Matrix must be square");

	Map<MatrixXd> eigen_mtx(mtx.matrix, mtx.num_rows, mtx.num_cols);
	LLT<MatrixXd> chol(eigen_mtx.colwise().reverse().rowwise().reverse().matrix());
	//tmp=chol(rot180(A))'
	MatrixXd tmp=chol.matrixL();
	SGMatrix<float64_t> res(mtx.num_rows, mtx.num_cols);
	Map<MatrixXd> eigen_res(res.matrix, res.num_rows, res.num_cols);
	//chol_inv = @(A) rot180(chol(rot180(A))')\eye(nu);                 % chol(inv(A))
	eigen_res=tmp.colwise().reverse().rowwise().reverse().matrix().triangularView<Upper>(
		).solve(MatrixXd::Identity(mtx.num_rows, mtx.num_cols));
	return res;
}

float64_t SingleFITCLaplaceInferenceMethod::get_negative_log_marginal_likelihood()
{
	if (parameter_hash_changed())
		update();

	if (m_Wneg)
	{
		io::warn("nlZ cannot be computed since W is too negative");
		//nlZ = NaN;
		return Math::INFTY;
	}
	//time complexity O(m^2*n)
	Map<VectorXd> eigen_alpha(m_al.vector, m_al.vlen);
	Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);
	// get log likelihood
	float64_t lp=SGVector<float64_t>::sum(m_model->get_log_probability_f(m_labels,
		m_mu));

	Map<MatrixXd> eigen_V(m_V.matrix, m_V.num_rows, m_V.num_cols);
	Map<VectorXd>eigen_t(m_t.vector, m_t.vlen);
	MatrixXd A=eigen_V*eigen_t.asDiagonal()*eigen_V.transpose()+MatrixXd::Identity(m_V.num_rows,m_V.num_rows);
	LLT<MatrixXd> chol(A);
	A=chol.matrixU();

	Map<VectorXd> eigen_dg(m_dg.vector, m_dg.vlen);
	Map<VectorXd> eigen_W(m_W.vector, m_W.vlen);

	//nlZ = alpha'*(f-m)/2 - sum(lp) - sum(log(dd))/2 + sum(log(diag(chol(A))));
	float64_t result=eigen_alpha.dot(eigen_mu-eigen_mean)/2.0-lp+
		A.diagonal().array().log().sum()+(eigen_W.cwiseProduct(eigen_dg)+MatrixXd::Ones(eigen_dg.rows(),1)).array().log().sum()/2.0;

	return result;
}

void SingleFITCLaplaceInferenceMethod::update_approx_cov()
{
}

void SingleFITCLaplaceInferenceMethod::update_init()
{
	//time complexity O(m^2*n)
	//m-by-m matrix
	Map<MatrixXd> eigen_kuu(m_kuu.matrix, m_kuu.num_rows, m_kuu.num_cols);
	//m-by-n matrix
	Map<MatrixXd> eigen_ktru(m_ktru.matrix, m_ktru.num_rows, m_ktru.num_cols);

	Map<VectorXd> eigen_ktrtr_diag(m_ktrtr_diag.vector, m_ktrtr_diag.vlen);

	SGMatrix<float64_t> cor_kuu(m_kuu.num_rows, m_kuu.num_cols);
	Map<MatrixXd> eigen_cor_kuu(cor_kuu.matrix, cor_kuu.num_rows, cor_kuu.num_cols);
	eigen_cor_kuu = eigen_kuu * std::exp(m_log_scale * 2.0) +
		            std::exp(m_log_ind_noise) *
		                MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols);
	//R0 = chol_inv(Kuu+snu2*eye(nu)); m-by-m matrix
	m_chol_R0=get_chol_inv(cor_kuu);
	Map<MatrixXd> eigen_R0(m_chol_R0.matrix, m_chol_R0.num_rows, m_chol_R0.num_cols);

	//V = R0*Ku;  m-by-n matrix
	m_V=SGMatrix<float64_t>(m_chol_R0.num_cols, m_ktru.num_cols);
	Map<MatrixXd> eigen_V(m_V.matrix, m_V.num_rows, m_V.num_cols);

	eigen_V = eigen_R0 * (eigen_ktru * std::exp(m_log_scale * 2.0));
	m_dg=SGVector<float64_t>(m_ktrtr_diag.vlen);
	Map<VectorXd> eigen_dg(m_dg.vector, m_dg.vlen);
	//d0 = diagK-sum(V.*V,1)';
	eigen_dg = eigen_ktrtr_diag * std::exp(m_log_scale * 2.0) -
		       (eigen_V.cwiseProduct(eigen_V)).colwise().sum().adjoint();

	// get mean vector and create eigen representation of it
	m_mean_f=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mean(m_mean_f.vector, m_mean_f.vlen);

	// create shogun and eigen representation of function vector
	m_mu=SGVector<float64_t>(m_mean_f.vlen);
	Map<VectorXd> eigen_mu(m_mu, m_mu.vlen);

	float64_t Psi_New;
	float64_t Psi_Def;
	if (m_al.vlen!=m_labels->get_num_labels())
	{
		// set alpha a zero vector
		m_al=SGVector<float64_t>(m_labels->get_num_labels());
		m_al.zero();

		// f = mean, if length of alpha and length of y doesn't match
		eigen_mu=eigen_mean;

		Psi_New=-SGVector<float64_t>::sum(m_model->get_log_probability_f(
			m_labels, m_mu));
	}
	else
	{
		Map<VectorXd> eigen_alpha(m_al.vector, m_al.vlen);

		// compute f = K * alpha + m
		SGVector<float64_t> tmp=compute_mvmK(m_al);
		Map<VectorXd> eigen_tmp(tmp.vector, tmp.vlen);
		eigen_mu=eigen_tmp+eigen_mean;

		Psi_New=eigen_alpha.dot(eigen_tmp)/2.0-
			SGVector<float64_t>::sum(m_model->get_log_probability_f(m_labels, m_mu));

		Psi_Def=-SGVector<float64_t>::sum(m_model->get_log_probability_f(m_labels, m_mean_f));

		// if default is better, then use it
		if (Psi_Def < Psi_New)
		{
			m_al.zero();
			eigen_mu=eigen_mean;
			Psi_New=Psi_Def;
		}
	}
	m_Psi=Psi_New;
}

void SingleFITCLaplaceInferenceMethod::register_minimizer(std::shared_ptr<Minimizer> minimizer)
{
	require(minimizer, "Minimizer must set");
	if (!std::dynamic_pointer_cast<SingleFITCLaplaceNewtonOptimizer>(minimizer))
	{
		auto opt= std::dynamic_pointer_cast<FirstOrderMinimizer>(minimizer);
		require(opt, "The provided minimizer is not supported");
	}
	Inference::register_minimizer(minimizer);
}


void SingleFITCLaplaceInferenceMethod::update_alpha()
{
	auto opt=std::dynamic_pointer_cast<SingleFITCLaplaceNewtonOptimizer>(m_minimizer);
	bool cleanup=false;
	if (opt)
	{
		opt->set_target(shared_from_this()->as<SingleFITCLaplaceInferenceMethod>());
		opt->minimize();
	}
	else
	{
		auto minimizer= std::dynamic_pointer_cast<FirstOrderMinimizer>(m_minimizer);
		require(minimizer, "The provided minimizer is not supported");

		auto cost_fun=std::make_shared<SingleFITCLaplaceInferenceMethodCostFunction>();
		cost_fun->set_target(shared_from_this()->as<SingleFITCLaplaceInferenceMethod>());
		minimizer->set_cost_function(cost_fun);
		minimizer->minimize();
		minimizer->unset_cost_function(false);

	}

	Map<VectorXd> eigen_mean(m_mean_f.vector, m_mean_f.vlen);
	Map<MatrixXd> eigen_V(m_V.matrix, m_V.num_rows, m_V.num_cols);
	Map<MatrixXd> eigen_R0(m_chol_R0.matrix, m_chol_R0.num_rows, m_chol_R0.num_cols);
	Map<VectorXd> eigen_mu(m_mu, m_mu.vlen);
	Map<VectorXd> eigen_al(m_al.vector, m_al.vlen);


	// compute f = K * alpha + m
	SGVector<float64_t> tmp=compute_mvmK(m_al);
	Map<VectorXd> eigen_tmp(tmp.vector, tmp.vlen);
	eigen_mu=eigen_tmp+eigen_mean;

	m_alpha=SGVector<float64_t>(m_chol_R0.num_cols);
	Map<VectorXd> eigen_post_alpha(m_alpha.vector, m_alpha.vlen);
	//post.alpha = R0'*(V*alpha);
	//m-by-1 vector
	eigen_post_alpha=eigen_R0.transpose()*(eigen_V*eigen_al);
}

void SingleFITCLaplaceInferenceMethod::update_chol()
{
	//time complexity O(m^2*n)
	Map<VectorXd> eigen_dg(m_dg.vector, m_dg.vlen);
	Map<MatrixXd> eigen_R0(m_chol_R0.matrix, m_chol_R0.num_rows, m_chol_R0.num_cols);
	Map<MatrixXd> eigen_V(m_V.matrix, m_V.num_rows, m_V.num_cols);

	// get log probability derivatives
	m_dlp=m_model->get_log_probability_derivative_f(m_labels, m_mu, 1);
	m_d2lp=m_model->get_log_probability_derivative_f(m_labels, m_mu, 2);
	m_d3lp=m_model->get_log_probability_derivative_f(m_labels, m_mu, 3);

	// W = -d2lp
	m_W=m_d2lp.clone();
	m_W.scale(-1.0);

	Map<VectorXd> eigen_W(m_W.vector, m_W.vlen);
	m_sW=SGVector<float64_t>(m_W.vlen);
	Map<VectorXd> eigen_sW(m_sW.vector, m_sW.vlen);

	VectorXd Wd0_1=eigen_W.cwiseProduct(eigen_dg)+MatrixXd::Ones(eigen_W.rows(),1);

	// compute sW
	// post.sW = sqrt(abs(W)).*sign(W);             % preserve sign in case of negative
	if (eigen_W.minCoeff()>0)
	{
		eigen_sW=eigen_W.cwiseSqrt();
	}
	else
	{
		eigen_sW=((eigen_W.array().abs()+eigen_W.array())/2).sqrt()-((eigen_W.array().abs()-eigen_W.array())/2).sqrt();
		//any(1+d0.*W<0)
		if (!(Wd0_1.array().abs().matrix()==Wd0_1))
			m_Wneg=true;
	}

	m_t=SGVector<float64_t>(m_W.vlen);
	Map<VectorXd>eigen_t(m_t.vector, m_t.vlen);

	//dd = 1./(1+d0.*W);
	VectorXd dd=MatrixXd::Ones(Wd0_1.rows(),1).cwiseQuotient(Wd0_1);
	eigen_t=eigen_W.cwiseProduct(dd);

	//m-by-m matrix
	SGMatrix<float64_t> A(m_V.num_rows, m_V.num_rows);
	Map<MatrixXd> eigen_A(A.matrix, A.num_rows, A.num_cols);
	//A = eye(nu)+(V.*repmat((W.*dd)',nu,1))*V';
	eigen_A=eigen_V*eigen_t.asDiagonal()*eigen_V.transpose()+MatrixXd::Identity(A.num_rows,A.num_rows);

	//R0tV = R0'*V; m-by-n
	MatrixXd R0tV=eigen_R0.transpose()*eigen_V;

	//B = R0tV.*repmat((W.*dd)',nu,1); m-by-n matrix
	MatrixXd B=R0tV*eigen_t.asDiagonal();

	//m-by-m matrix
	m_L=SGMatrix<float64_t>(m_kuu.num_rows, m_kuu.num_cols);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	//post.L = -B*R0tV';
	eigen_L=-B*R0tV.transpose();

	SGMatrix<float64_t> tmp=get_chol_inv(A);
	Map<MatrixXd> eigen_tmp(tmp.matrix, tmp.num_rows, tmp.num_cols);
	//RV = chol_inv(A)*V; m-by-n matrix
	MatrixXd eigen_RV=eigen_tmp*eigen_V;
	//RVdd m-by-n matrix
	m_Rvdd=SGMatrix<float64_t>(m_V.num_rows, m_V.num_cols);
	Map<MatrixXd> eigen_Rvdd(m_Rvdd.matrix, m_Rvdd.num_rows, m_Rvdd.num_cols);
	//RVdd = RV.*repmat((W.*dd)',nu,1);
	eigen_Rvdd=eigen_RV*eigen_t.asDiagonal();

	if (!m_Wneg)
	{
		//B = B*RV';
		B=B*eigen_RV.transpose();
		//post.L = post.L + B*B';
		eigen_L+=B*B.transpose();
	}
	else
	{
		//B = B*V';
		B=B*eigen_V.transpose();
		//post.L = post.L + (B*inv(A))*B';
		FullPivLU<MatrixXd> lu(eigen_A);
		eigen_L+=B*lu.inverse()*B.transpose();
	}

	Map<MatrixXd> eigen_ktru(m_ktru.matrix, m_ktru.num_rows, m_ktru.num_cols);
	m_g=SGVector<float64_t>(m_dg.vlen);
	Map<VectorXd> eigen_g(m_g.vector, m_g.vlen);
	//g = d/2 + sum(((R*R0)*P).^2,1)'/2
	eigen_g = ((eigen_dg.cwiseProduct(dd)).array() +
		       ((eigen_tmp * eigen_R0) *
		        (eigen_ktru * std::exp(m_log_scale * 2.0)) * dd.asDiagonal())
		           .array()
		           .pow(2)
		           .colwise()
		           .sum()
		           .transpose()) /
		      2;
}

void SingleFITCLaplaceInferenceMethod::update_deriv()
{
	//time complexity O(m^2*n)
	Map<MatrixXd> eigen_ktru(m_ktru.matrix, m_ktru.num_rows, m_ktru.num_cols);
	Map<MatrixXd> eigen_R0(m_chol_R0.matrix, m_chol_R0.num_rows, m_chol_R0.num_cols);
	Map<VectorXd> eigen_al(m_al.vector, m_al.vlen);
	Map<MatrixXd> eigen_V(m_V.matrix, m_V.num_rows, m_V.num_cols);
	// create shogun and eigen representation of B
	// m-by-n matrix
	m_B=SGMatrix<float64_t>(m_ktru.num_rows, m_ktru.num_cols);
	Map<MatrixXd> eigen_B(m_B.matrix, m_B.num_rows, m_B.num_cols);

	//B = (R0'*R0)*Ku
	eigen_B=eigen_R0.transpose()*eigen_V;

	// create shogun and eigen representation of w
	m_w=SGVector<float64_t>(m_B.num_rows);
	//w = B*al;
	Map<VectorXd> eigen_w(m_w.vector, m_w.vlen);
	eigen_w=eigen_B*eigen_al;

	// create shogun and eigen representation of the vector dfhat
	Map<VectorXd> eigen_d3lp(m_d3lp.vector, m_d3lp.vlen);
	Map<VectorXd> eigen_g(m_g.vector, m_g.vlen);
	m_dfhat=SGVector<float64_t>(m_g.vlen);
	Map<VectorXd> eigen_dfhat(m_dfhat.vector, m_dfhat.vlen);

	// compute derivative of nlZ wrt fhat
	// dfhat = g.*d3lp;
	eigen_dfhat=eigen_g.cwiseProduct(eigen_d3lp);
}

float64_t SingleFITCLaplaceInferenceMethod::get_derivative_related_cov(SGVector<float64_t> ddiagKi,
	SGMatrix<float64_t> dKuui, SGMatrix<float64_t> dKui)
{
	//time complexity O(m^2*n)
	Map<MatrixXd> eigen_R0tV(m_B.matrix, m_B.num_rows, m_B.num_cols);
	Map<VectorXd> eigen_ddiagKi(ddiagKi.vector, ddiagKi.vlen);
	//m-by-m matrix
	Map<MatrixXd> eigen_dKuui(dKuui.matrix, dKuui.num_rows, dKuui.num_cols);
	//m-by-n matrix
	Map<MatrixXd> eigen_dKui(dKui.matrix, dKui.num_rows, dKui.num_cols);

	// compute R=2*dKui-dKuui*B
	SGMatrix<float64_t> dA(dKui.num_rows, dKui.num_cols);
	Map<MatrixXd> eigen_dA(dA.matrix, dA.num_rows, dA.num_cols);
	//dA = 2*dKu'-R0tV'*dKuu;
	//dA' = 2*dKu-dKuu'*R0tV;
	eigen_dA=2*eigen_dKui-eigen_dKuui*eigen_R0tV;

	SGVector<float64_t> v(ddiagKi.vlen);
	Map<VectorXd> eigen_v(v.vector, v.vlen);
	//w = sum(dA.*R0tV',2);
	//w' = sum(dA'.*R0tV,1);
	//v = ddiagK-w;
	eigen_v=eigen_ddiagKi-eigen_dA.cwiseProduct(eigen_R0tV).colwise().sum().transpose();

	//explicit term
	float64_t result=SingleFITCInference::get_derivative_related_cov(ddiagKi, dKuui, dKui, v, dA);

	//implicit term
	Map<VectorXd> eigen_dlp(m_dlp.vector, m_dlp.vlen);
	Map<VectorXd> eigen_dfhat(m_dfhat.vector, m_dfhat.vlen);

	SGVector<float64_t> b(v.vlen);
	Map<VectorXd> eigen_b(b.vector, b.vlen);
	//b = dA*(R0tV*dlp) + v.*dlp;
	eigen_b=eigen_dA.transpose()*(eigen_R0tV*eigen_dlp)+eigen_v.cwiseProduct(eigen_dlp);
	//KZb = mvmK(mvmZ(b,RVdd,t),V,d0);
	SGVector<float64_t> KZb=compute_mvmK(compute_mvmZ(b));
	Map<VectorXd> eigen_KZb(KZb.vector, KZb.vlen);
	//dnlZ.cov(i) = dnlZ.cov(i) - dfhat'*( b-KZb );
	result-=eigen_dfhat.dot(eigen_b-eigen_KZb);
	return result;
}

SGVector<float64_t> SingleFITCLaplaceInferenceMethod::get_derivative_wrt_inference_method(
		Parameters::const_reference param)
{
	//time complexity O(m^2*n);
	require(param.first == "log_scale"
		|| param.first == "log_inducing_noise"
		|| param.first == "inducing_features",
		"Can't compute derivative of"
		" the nagative log marginal likelihood wrt {}.{} parameter",
		get_name(), param.first);

	SGVector<float64_t> result;
	int32_t len;
	if (param.first == "inducing_features")
	{
		if(m_Wneg)
		{
			auto inducing_feat = m_inducing_features->as<DotFeatures>();
			int32_t dim = inducing_feat->get_dim_feature_space();
			int32_t num_samples = inducing_feat->get_num_vectors();
			len=dim*num_samples;
		}
		else if (!m_fully_sparse)
			return SingleFITCInference::get_derivative_wrt_inference_method(param);
		else
			return get_derivative_wrt_inducing_features(param);
	}
	else
		len=1;

	if (m_Wneg)
	{
		result=SGVector<float64_t>(len);
		return derivative_helper_when_Wneg(result, param);
	}

	if (param.first == "log_inducing_noise")
		// wrt inducing_noise
		// compute derivative wrt inducing noise
		return get_derivative_wrt_inducing_noise(param);

	result=SGVector<float64_t>(len);
	// wrt scale
	// clone kernel matrices
	SGVector<float64_t> deriv_trtr=m_ktrtr_diag.clone();
	SGMatrix<float64_t> deriv_uu=m_kuu.clone();
	SGMatrix<float64_t> deriv_tru=m_ktru.clone();

	// create eigen representation of kernel matrices
	Map<VectorXd> ddiagKi(deriv_trtr.vector, deriv_trtr.vlen);
	Map<MatrixXd> dKuui(deriv_uu.matrix, deriv_uu.num_rows, deriv_uu.num_cols);
	Map<MatrixXd> dKui(deriv_tru.matrix, deriv_tru.num_rows, deriv_tru.num_cols);

	// compute derivatives wrt scale for each kernel matrix
	result[0]=get_derivative_related_cov(deriv_trtr, deriv_uu, deriv_tru);
	result[0] *= std::exp(m_log_scale * 2.0) * 2.0;
	return result;
}

SGVector<float64_t> SingleFITCLaplaceInferenceMethod::get_derivative_wrt_likelihood_model(
		Parameters::const_reference param)
{
	SGVector<float64_t> result(1);
	if (m_Wneg)
		return derivative_helper_when_Wneg(result, param);

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
	Map<VectorXd> eigen_g(m_g.vector, m_g.vlen);
	Map<VectorXd> eigen_dfhat(m_dfhat.vector, m_dfhat.vlen);

	//explicit term
	//dnlZ.lik(i) = -g'*d2lp_dhyp - sum(lp_dhyp);
	result[0]=-eigen_g.dot(eigen_d2lp_dhyp)-eigen_lp_dhyp.sum();

	//implicit term
	//b = mvmK(dlp_dhyp,V,d0);
	SGVector<float64_t> b=compute_mvmK(dlp_dhyp);
	//dnlZ.lik(i) = dnlZ.lik(i) - dfhat'*(b-mvmK(mvmZ(b,RVdd,t),V,d0));
	result[0]-= get_derivative_implicit_term_helper(b);

	return result;
}

SGVector<float64_t> SingleFITCLaplaceInferenceMethod::get_derivative_wrt_kernel(
		Parameters::const_reference param)
{
	SGVector<float64_t> result;
	auto visitor = std::make_unique<ShapeVisitor>();
	param.second->get_value().visit(visitor.get());
	int64_t len= visitor->get_size();
	result=SGVector<float64_t>(len);

	if (m_Wneg)
		return derivative_helper_when_Wneg(result, param);

	m_lock.lock();
	auto inducing_features=get_inducing_features();
	for (index_t i=0; i<result.vlen; i++)
	{
		//time complexity O(m^2*n)
		SGVector<float64_t> deriv_trtr;
		SGMatrix<float64_t> deriv_uu;
		SGMatrix<float64_t> deriv_tru;

		m_kernel->init(m_features, m_features);
		deriv_trtr=m_kernel->get_parameter_gradient_diagonal(param, i);

		m_kernel->init(inducing_features, inducing_features);
		deriv_uu=m_kernel->get_parameter_gradient(param, i);

		m_kernel->init(inducing_features, m_features);
		deriv_tru=m_kernel->get_parameter_gradient(param, i);

		// create eigen representation of derivatives
		Map<VectorXd> ddiagKi(deriv_trtr.vector, deriv_trtr.vlen);
		Map<MatrixXd> dKuui(deriv_uu.matrix, deriv_uu.num_rows,
				deriv_uu.num_cols);
		Map<MatrixXd> dKui(deriv_tru.matrix, deriv_tru.num_rows,
				deriv_tru.num_cols);

		result[i]=get_derivative_related_cov(deriv_trtr, deriv_uu, deriv_tru);
		result[i] *= std::exp(m_log_scale * 2.0);
	}

	m_lock.unlock();

	return result;
}

float64_t SingleFITCLaplaceInferenceMethod::get_derivative_related_mean(SGVector<float64_t> dmu)
{
	//time complexity O(m*n)
	//explicit term
	float64_t result=SingleFITCInference::get_derivative_related_mean(dmu);

	//implicit term
	//Zdm = mvmZ(dm,RVdd,t);
	//tmp = mvmK(Zdm,V,d0)
	//dnlZ.mean(i) = dnlZ.mean(i) - dfhat'*(dm-mvmK(Zdm,V,d0));
	result-=get_derivative_implicit_term_helper(dmu);

	return result;
}

float64_t SingleFITCLaplaceInferenceMethod::get_derivative_implicit_term_helper(SGVector<float64_t> d)
{
	//time complexity O(m*n)
	Map<VectorXd> eigen_d(d.vector, d.vlen);
	SGVector<float64_t> tmp=compute_mvmK(compute_mvmZ(d));
	Map<VectorXd> eigen_tmp(tmp.vector, tmp.vlen);
	Map<VectorXd> eigen_dfhat(m_dfhat.vector, m_dfhat.vlen);
	return eigen_dfhat.dot(eigen_d-eigen_tmp);
}

SGVector<float64_t> SingleFITCLaplaceInferenceMethod::get_derivative_wrt_mean(
		Parameters::const_reference param)
{
	//time complexity O(m*n)
	SGVector<float64_t> result;
	auto visitor = std::make_unique<ShapeVisitor>();
	param.second->get_value().visit(visitor.get());
	int64_t len= visitor->get_size();
	result=SGVector<float64_t>(len);

	if (m_Wneg)
		return derivative_helper_when_Wneg(result, param);

	for (index_t i=0; i<result.vlen; i++)
	{
		SGVector<float64_t> dmu;
		dmu=m_mean->get_parameter_derivative(m_features, param, i);
		result[i]=get_derivative_related_mean(dmu);
	}

	return result;
}

SGVector<float64_t> SingleFITCLaplaceInferenceMethod::derivative_helper_when_Wneg(
	SGVector<float64_t> res, Parameters::const_reference param)
{
	io::warn("Derivative wrt {} cannot be computed since W (the Hessian (diagonal) matrix) is too negative", param.first);
	//dnlZ = struct('cov',0*hyp.cov, 'mean',0*hyp.mean, 'lik',0*hyp.lik);
	res.zero();
	return res;
}

SGVector<float64_t> SingleFITCLaplaceInferenceMethod::get_derivative_wrt_inducing_features(
	Parameters::const_reference param)
{
	//time complexity depends on the implementation of the provided kernel
	//time complexity is at least O(max((p*n*m),(m^2*n))), where p is the dimension (#) of features
	//For an ARD kernel with KL_FULL, the time complexity is O(max((p*n*m*d),(m^2*n)))
	//where the paramter \f$\Lambda\f$ of the ARD kerenl is a \f$d\f$-by-\f$p\f$ matrix,
	//For an ARD kernel with KL_SCALE and KL_DIAG, the time complexity is O(max((p*n*m),(m^2*n)))
	//efficiently compute the implicit term and explicit term at one shot
	Map<VectorXd> eigen_al(m_al.vector, m_al.vlen);
	Map<MatrixXd> eigen_Rvdd(m_Rvdd.matrix, m_Rvdd.num_rows, m_Rvdd.num_cols);
	//w=B*al
	Map<VectorXd> eigen_w(m_w.vector, m_w.vlen);
	Map<MatrixXd> eigen_B(m_B.matrix, m_B.num_rows, m_B.num_cols);
	Map<VectorXd> eigen_dlp(m_dlp.vector, m_dlp.vlen);
	Map<VectorXd> eigen_dfhat(m_dfhat.vector, m_dfhat.vlen);

	//q = dfhat - mvmZ(mvmK(dfhat,V,d0),RVdd,t);
	SGVector<float64_t> q=compute_mvmZ(compute_mvmK(m_dfhat));
	Map<VectorXd> eigen_q(q.vector, q.vlen);
	eigen_q=eigen_dfhat-eigen_q;

	//explicit term
	//diag_dK = alpha.*alpha + sum(RVdd.*RVdd,1)'-t, where t can be cancelled out
	//-v_1=get_derivative_related_cov_diagonal= -(alpha.*alpha + sum(RVdd.*RVdd,1)')
	//implicit term
	//-v_2=-2*dlp.*q
	//neg_v = -(diag_dK+ 2*dlp.*q);
	SGVector<float64_t> neg_v=get_derivative_related_cov_diagonal();
	Map<VectorXd> eigen_neg_v(neg_v.vector, neg_v.vlen);
	eigen_neg_v-=2*eigen_dlp.cwiseProduct(eigen_q);

	SGMatrix<float64_t> BdK(m_B.num_rows, m_B.num_cols);
	Map<MatrixXd> eigen_BdK(BdK.matrix, BdK.num_rows, BdK.num_cols);
	//explicit
	//BdK = (B*alpha)*alpha' + (B*RVdd')*RVdd - B.*repmat(v_1',nu,1),
	//implicit
	//BdK = BdK + (B*dlp)*q' + (B*q)*dlp' - B.*repmat(v_2',nu,1)
	//where v_1 is the explicit part of v and v_2 is the implicit part of v
	//v=v_1+v_2
	eigen_BdK=eigen_B*eigen_neg_v.asDiagonal()+eigen_w*(eigen_al.transpose())+
		(eigen_B*eigen_Rvdd.transpose())*eigen_Rvdd+
		(eigen_B*eigen_dlp)*eigen_q.transpose()+(eigen_B*eigen_q)*eigen_dlp.transpose();

	return get_derivative_related_inducing_features(BdK, param);
}

SGVector<float64_t> SingleFITCLaplaceInferenceMethod::get_derivative_wrt_inducing_noise(
	Parameters::const_reference param)
{
	//time complexity O(m^2*n)
	//explicit term
	SGVector<float64_t> result=SingleFITCInference::get_derivative_wrt_inducing_noise(param);

	//implicit term
	Map<MatrixXd> eigen_B(m_B.matrix, m_B.num_rows, m_B.num_cols);
	Map<VectorXd> eigen_dlp(m_dlp.vector, m_dlp.vlen);

	//snu = sqrt(snu2);
	//T = chol_inv(Kuu + snu2*eye(nu)); T = T'*(T*(snu*Ku));
	//t1 = sum(T.*T,1)';
	VectorXd eigen_t1=eigen_B.cwiseProduct(eigen_B).colwise().sum().adjoint();

	//b = (t1.*dlp-T'*(T*dlp))*2;
	SGVector<float64_t> b(eigen_t1.rows());
	Map<VectorXd> eigen_b(b.vector, b.vlen);
	float64_t factor = 2.0 * std::exp(m_log_ind_noise);
	eigen_b=(eigen_t1.cwiseProduct(eigen_dlp)-eigen_B.transpose()*(eigen_B*eigen_dlp))*factor;

	//KZb = mvmK(mvmZ(b,RVdd,t),V,d0);
	//z = z - dfhat'*( b-KZb );
	result[0]-=get_derivative_implicit_term_helper(b);

	return result;
}

SGVector<float64_t> SingleFITCLaplaceInferenceMethod::get_posterior_mean()
{
	compute_gradient();

	SGVector<float64_t> res(m_mu.vlen);
	Map<VectorXd> eigen_res(res.vector, res.vlen);

	/*
	//true posterior mean with equivalent FITC prior approximated by Newton method
	//time complexity O(n)
	Map<VectorXd> eigen_mu(m_mu, m_mu.vlen);
	SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);
	eigen_res=eigen_mu-eigen_mean;
	*/

	//FITC (further) approximated posterior mean with Netwon method
	//time complexity of the following operation is O(m*n)
	Map<VectorXd> eigen_post_alpha(m_alpha.vector, m_alpha.vlen);
	Map<MatrixXd> eigen_Ktru(m_ktru.matrix, m_ktru.num_rows, m_ktru.num_cols);
	eigen_res =
		std::exp(m_log_scale * 2.0) * eigen_Ktru.adjoint() * eigen_post_alpha;

	return res;
}

SGMatrix<float64_t> SingleFITCLaplaceInferenceMethod::get_posterior_covariance()
{
	compute_gradient();
	//time complexity of the following operations is O(m*n^2)
	//Warning: the the time complexity increases from O(m^2*n) to O(n^2*m) if this method is called
	m_Sigma=SGMatrix<float64_t>(m_ktrtr_diag.vlen, m_ktrtr_diag.vlen);
	Map<MatrixXd> eigen_Sigma(m_Sigma.matrix, m_Sigma.num_rows,
			m_Sigma.num_cols);

	//FITC (further) approximated posterior covariance with Netwon method
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_Ktru(m_ktru.matrix, m_ktru.num_rows, m_ktru.num_cols);
	Map<VectorXd> eigen_dg(m_dg.vector, m_dg.vlen);
	Map<MatrixXd> eigen_V(m_V.matrix, m_V.num_rows, m_V.num_cols);

	MatrixXd diagonal_part=eigen_dg.asDiagonal();
	//FITC equivalent prior
	MatrixXd prior=eigen_V.transpose()*eigen_V+diagonal_part;

	MatrixXd tmp = std::exp(m_log_scale * 2.0) * eigen_Ktru;
	eigen_Sigma=prior-tmp.adjoint()*eigen_L*tmp;

	/*
	//true posterior mean with equivalent FITC prior approximated by Newton method
	Map<VectorXd> eigen_t(m_t.vector, m_t.vlen);
	Map<MatrixXd> eigen_Rvdd(m_Rvdd.matrix, m_Rvdd.num_rows, m_Rvdd.num_cols);
	Map<VectorXd> eigen_W(m_W.vector, m_W.vlen);
	MatrixXd tmp1=eigen_Rvdd*prior;
	eigen_Sigma=prior+tmp.transpose()*tmp;

	MatrixXd tmp2=((eigen_dg.cwiseProduct(eigen_t)).asDiagonal()*eigen_V.transpose())*eigen_V;
	eigen_Sigma-=(tmp2+tmp2.transpose());
	eigen_Sigma-=eigen_V.transpose()*(eigen_V*eigen_t.asDiagonal()*eigen_V.transpose())*eigen_V;
	MatrixXd tmp3=((eigen_dg.cwiseProduct(eigen_dg)).cwiseProduct(eigen_t)).asDiagonal();
	eigen_Sigma-=tmp3;
	*/

	return SGMatrix<float64_t>(m_Sigma);
}

float64_t SingleFITCLaplaceInferenceMethod::get_psi_wrt_alpha()
{
	//time complexity O(m*n)
	Map<VectorXd> eigen_alpha(m_al, m_al.vlen);
	SGVector<float64_t> f(m_al.vlen);
	Map<VectorXd> eigen_f(f.vector, f.vlen);
	Map<VectorXd> eigen_mean_f(m_mean_f.vector,m_mean_f.vlen);
	/* f = K * alpha + mean_f given alpha*/
	SGVector<float64_t> tmp=compute_mvmK(m_al);
	Map<VectorXd> eigen_tmp(tmp.vector, tmp.vlen);
	eigen_f=eigen_tmp+eigen_mean_f;

	/* psi = 0.5 * alpha .* (f - m) - sum(dlp)*/
	float64_t psi=eigen_alpha.dot(eigen_tmp) * 0.5;
	psi-=SGVector<float64_t>::sum(m_model->get_log_probability_f(m_labels, f));

	return psi;
}

void SingleFITCLaplaceInferenceMethod::get_gradient_wrt_alpha(SGVector<float64_t> gradient)
{
	//time complexity O(m*n)
	Map<VectorXd> eigen_alpha(m_al, m_al.vlen);
	Map<VectorXd> eigen_gradient(gradient.vector, gradient.vlen);
	SGVector<float64_t> f(m_al.vlen);
	Map<VectorXd> eigen_f(f.vector, f.vlen);
	Map<MatrixXd> kernel(m_ktrtr.matrix,
		m_ktrtr.num_rows,
		m_ktrtr.num_cols);
	Map<VectorXd> eigen_mean_f(m_mean_f.vector, m_mean_f.vlen);

	/* f = K * alpha + mean_f given alpha*/
	SGVector<float64_t> tmp=compute_mvmK(m_al);
	Map<VectorXd> eigen_tmp(tmp.vector, tmp.vlen);
	eigen_f=eigen_tmp+eigen_mean_f;

	SGVector<float64_t> dlp_f =
		m_model->get_log_probability_derivative_f(m_labels, f, 1);

	Map<VectorXd> eigen_dlp_f(dlp_f.vector, dlp_f.vlen);

	/* g_alpha = K * (alpha - dlp_f)*/
	SGVector<float64_t> tmp2(m_al.vlen);
	Map<VectorXd> eigen_tmp2(tmp2.vector, tmp2.vlen);
	eigen_tmp2=eigen_alpha-eigen_dlp_f;
	tmp2=compute_mvmK(tmp2);
	Map<VectorXd> eigen_tmp3(tmp2.vector, tmp2.vlen);
	eigen_gradient=eigen_tmp3;
}

}
