/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 *
 * Code adapted from Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * This code specifically adapted from infLaplace.m
 */

#include <shogun/machine/gp/LaplacianInferenceMethod.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/StudentsTLikelihood.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/features/DotFeatures.h>

#include <shogun/lib/external/brent.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/** Wrapper class used for the Brent minimizer */
class CPsiLine : public func_base
{
public:
	float64_t scale;
	MatrixXd K;
	VectorXd dalpha;
	VectorXd start_alpha;
	Map<VectorXd>* alpha;
	SGVector<float64_t>* dlp;
	SGVector<float64_t>* W;
	SGVector<float64_t>* f;
	SGVector<float64_t>* m;
	CLikelihoodModel* lik;
	CLabels* lab;

	virtual double operator() (double x)
	{
		Map<VectorXd> eigen_f(f->vector, f->vlen);
		Map<VectorXd> eigen_m(m->vector, m->vlen);

		// compute alpha=alpha+x*dalpha and f=K*alpha+m
		(*alpha)=start_alpha+x*dalpha;
		eigen_f=K*(*alpha)*CMath::sq(scale)+eigen_m;

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

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

CLaplacianInferenceMethod::CLaplacianInferenceMethod() : CInferenceMethod()
{
	init();
}

CLaplacianInferenceMethod::CLaplacianInferenceMethod(CKernel* kern,
		CFeatures* feat, CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod)
		: CInferenceMethod(kern, feat, m, lab, mod)
{
	init();
}

void CLaplacianInferenceMethod::init()
{
	m_iter=20;
	m_tolerance=1e-6;
	m_opt_tolerance=1e-10;
	m_opt_max=10;
}

CLaplacianInferenceMethod::~CLaplacianInferenceMethod()
{
}

void CLaplacianInferenceMethod::update_all()
{
	check_members();

	// update feature matrix
	CFeatures* feat=m_features;

	if (m_features->get_feature_class()==C_COMBINED)
		feat=((CCombinedFeatures*)m_features)->get_first_feature_obj();
	else
		SG_REF(m_features);

	m_feature_matrix=((CDotFeatures*)feat)->get_computed_dot_feature_matrix();

	SG_UNREF(feat);

	update_train_kernel();
	update_alpha();
	update_chol();
	update_approx_cov();
}

CMap<TParameter*, SGVector<float64_t> > CLaplacianInferenceMethod::
get_marginal_likelihood_derivatives(CMap<TParameter*, CSGObject*>& para_dict)
{
	if (update_parameter_hash())
		update_all();

	// create eigen representation of W, sW, dlp, d3lp, K, alpha and L
	Map<VectorXd> eigen_W(W.vector, W.vlen);
	Map<VectorXd> eigen_sW(sW.vector, sW.vlen);
	Map<VectorXd> eigen_dlp(dlp.vector, dlp.vlen);
	Map<VectorXd> eigen_d3lp(d3lp.vector, d3lp.vlen);
	Map<MatrixXd> eigen_ktrtr(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	MatrixXd Z;
	VectorXd g;

	if (eigen_W.minCoeff() < 0)
	{
		Z=-eigen_L;

		// compute iA = (I + K * diag(W))^-1
		FullPivLU<MatrixXd> lu(MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols)+
			eigen_ktrtr*CMath::sq(m_scale)*eigen_W.asDiagonal());
		MatrixXd iA=lu.inverse();

		// compute derivative ln|L'*L| wrt W: g=sum(iA.*K,2)/2
		g=(iA.cwiseProduct(eigen_ktrtr*CMath::sq(m_scale))).rowwise().sum()/2.0;
	}
	else
	{
		// solve L'*L*Z=diag(sW) and compute Z=diag(sW)*Z
		Z=eigen_L.triangularView<Upper>().adjoint().solve(
			MatrixXd(eigen_sW.asDiagonal()));
		Z=eigen_L.triangularView<Upper>().solve(Z);
		Z=eigen_sW.asDiagonal()*Z;

		// solve L'*C=diag(sW)*K
		MatrixXd C=eigen_L.triangularView<Upper>().adjoint().solve(
			eigen_sW.asDiagonal()*eigen_ktrtr*CMath::sq(m_scale));

		// compute derivative ln|L'*L| wrt W: g=(diag(K)-sum(C.^2,1)')/2
		VectorXd sum=(C.cwiseProduct(C)).colwise().sum();
		g=(eigen_ktrtr.diagonal()*CMath::sq(m_scale)-sum)/2.0;
	}

	// compute derivative of nlZ wrt fhat
	VectorXd dfhat=g.cwiseProduct(eigen_d3lp);

	m_kernel->build_parameter_dictionary(para_dict);
	m_mean->build_parameter_dictionary(para_dict);
	m_model->build_parameter_dictionary(para_dict);

	CMap<TParameter*, SGVector<float64_t> > gradient(
			3+para_dict.get_num_elements(),
			3+para_dict.get_num_elements());

	for (index_t i = 0; i < para_dict.get_num_elements(); i++)
	{
		CMapNode<TParameter*, CSGObject*>* node=para_dict.get_node_ptr(i);

		TParameter* param = node->key;
		CSGObject* obj = node->data;

		index_t length = 1;

		if ((param->m_datatype.m_ctype== CT_VECTOR ||
				param->m_datatype.m_ctype == CT_SGVECTOR) &&
				param->m_datatype.m_length_y != NULL)
			length = *(param->m_datatype.m_length_y);

		SGVector<float64_t> variables(length);

		bool deriv_found = false;

		for (index_t h = 0; h < length; h++)
		{
			SGMatrix<float64_t> deriv=m_kernel->get_parameter_gradient(param, obj);
			SGVector<float64_t> lik_first_deriv=m_model->get_first_derivative(
				m_labels, param, m_approx_mean);
			SGVector<float64_t> lik_second_deriv=m_model->get_second_derivative(
				m_labels, param, m_approx_mean);
			SGVector<float64_t> lik_third_deriv=m_model->get_third_derivative(
				m_labels, param, m_approx_mean);
			SGVector<float64_t> mean_deriv;

			if (param->m_datatype.m_ctype==CT_VECTOR ||
					param->m_datatype.m_ctype==CT_SGVECTOR)
			{
				mean_deriv=m_mean->get_parameter_derivative(param, obj,
					m_feature_matrix, h);
			}
			else
			{
				mean_deriv=m_mean->get_parameter_derivative(param, obj,
					m_feature_matrix);
			}

			Map<VectorXd> eigen_mean_deriv(mean_deriv.vector, mean_deriv.vlen);

			if (deriv.num_cols*deriv.num_rows > 0)
			{
				Map<MatrixXd> dK(deriv.matrix, deriv.num_cols, deriv.num_rows);

				// compute dnlZ=sum(sum(Z.*dK))/2-alpha'*dK*alpha/2
				variables[h]=(Z.cwiseProduct(dK)).sum()/2.0-
					(eigen_alpha.transpose()*dK).dot(eigen_alpha)/2.0;

				// compute b=dK*dlp
				VectorXd b = dK*eigen_dlp;

				// compute dnlZ=dnlZ-dfhat'*(b-K*(Z*b))
				variables[h]=variables[h]-dfhat.dot(b-eigen_ktrtr*(Z*b)*CMath::sq(m_scale));
				deriv_found = true;
			}
			else if (mean_deriv.vlen > 0)
			{
				// compute dnlZ=-alpha'*dm-dfhat'*(dm-K*(Z*dm))
				variables[h]=-eigen_alpha.dot(eigen_mean_deriv)-dfhat.dot(
					eigen_mean_deriv-eigen_ktrtr*(Z*eigen_mean_deriv)*CMath::sq(m_scale));
				deriv_found = true;
			}
			else if (lik_first_deriv.vlen && lik_second_deriv.vlen && lik_third_deriv.vlen)
			{
				Map<VectorXd> eigen_fd(lik_first_deriv.vector, lik_first_deriv.vlen);
				Map<VectorXd> eigen_sd(lik_second_deriv.vector, lik_second_deriv.vlen);
				Map<VectorXd> eigen_td(lik_third_deriv.vector, lik_third_deriv.vlen);

				VectorXd b=eigen_ktrtr*eigen_sd;

				// compute dnlZ=-g'*d2lp_dhyp-sum(lp_dhyp)-dfhat'*(b-K*(Z*b))
				variables[h]=-g.dot(eigen_td)-eigen_fd.sum()-
					dfhat.dot(b-eigen_ktrtr*(Z*b)*CMath::sq(m_scale));
				deriv_found = true;
			}
		}

		if (deriv_found)
			gradient.add(param, variables);
	}

	TParameter* param;
	index_t index = get_modsel_param_index("scale");
	param = m_model_selection_parameters->get_parameter(index);

	SGVector<float64_t> scale(1);

	// compute derivative K wrt scale
	MatrixXd dK=eigen_ktrtr*m_scale*2.0;

	// compute dnlZ=sum(sum(Z.*dK))/2-alpha'*dK*alpha/2
	scale[0]=(Z.cwiseProduct(dK)).sum()/2.0-
		(eigen_alpha.transpose()*dK).dot(eigen_alpha)/2.0;

	// compute b=dK*dlp
	VectorXd b=dK*eigen_dlp;

	// compute dnlZ=dnlZ-dfhat'*(b-K*(Z*b))
	scale[0]=scale[0]-dfhat.transpose()*(b-eigen_ktrtr*(Z*b)*CMath::sq(m_scale));

	gradient.add(param, scale);
	para_dict.add(param, this);

	return gradient;
}

SGVector<float64_t> CLaplacianInferenceMethod::get_diagonal_vector()
{
	if (update_parameter_hash())
		update_all();

	SGVector<float64_t> result(sW);
	return result;
}

float64_t CLaplacianInferenceMethod::get_negative_marginal_likelihood()
{
	if (update_parameter_hash())
		update_all();

	// create eigen representations alpha, f, W, L
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	Map<VectorXd> eigen_approx_mean(m_approx_mean.vector, m_approx_mean.vlen);
	Map<VectorXd> eigen_W(W.vector, W.vlen);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	// get mean vector and create eigen representation of it
	SGVector<float64_t> mean=m_mean->get_mean_vector(m_feature_matrix);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);

	// get log likelihood
	float64_t lp=SGVector<float64_t>::sum(m_model->get_log_probability_f(m_labels,
		m_approx_mean));

	float64_t result;

	if (eigen_W.minCoeff()<0)
	{
		Map<VectorXd> eigen_sW(sW.vector, sW.vlen);
		Map<MatrixXd> eigen_ktrtr(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

		FullPivLU<MatrixXd> lu(MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols)+
			eigen_ktrtr*CMath::sq(m_scale)*eigen_sW.asDiagonal());

		result=(eigen_alpha.dot(eigen_approx_mean-eigen_mean))/2.0-
			lp+log(lu.determinant())/2.0;
	}
	else
	{
		result=eigen_alpha.dot(eigen_approx_mean-eigen_mean)/2.0-lp+
			eigen_L.diagonal().array().log().sum();
	}

	return result;
}

SGVector<float64_t> CLaplacianInferenceMethod::get_alpha()
{
	if (update_parameter_hash())
		update_all();

	SGVector<float64_t> result(m_alpha);
	return result;
}

SGMatrix<float64_t> CLaplacianInferenceMethod::get_cholesky()
{
	if (update_parameter_hash())
		update_all();

	return SGMatrix<float64_t>(m_L);
}

SGVector<float64_t> CLaplacianInferenceMethod::get_posterior_approximation_mean()
{
	if (update_parameter_hash())
		update_all();

	return SGVector<float64_t>(m_approx_mean);
}

SGMatrix<float64_t> CLaplacianInferenceMethod::get_posterior_approximation_covariance()
{
	if (update_parameter_hash())
		update_all();

	return SGMatrix<float64_t>(m_approx_cov);
}

void CLaplacianInferenceMethod::update_train_kernel()
{
	m_kernel->cleanup();
	m_kernel->init(m_features, m_features);
	m_ktrtr=m_kernel->get_kernel_matrix();
}

void CLaplacianInferenceMethod::update_approx_cov()
{
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> eigen_sW(sW.vector, sW.vlen);

	m_approx_cov=SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_approx_cov(m_approx_cov.matrix, m_approx_cov.num_rows,
		m_approx_cov.num_cols);

	MatrixXd eigen_iB=eigen_L.triangularView<Upper>().adjoint().solve(
		MatrixXd::Identity(m_L.num_rows, m_L.num_cols));
	eigen_iB=eigen_L.triangularView<Upper>().solve(eigen_iB);

	eigen_approx_cov=eigen_K-eigen_K*eigen_sW.asDiagonal()*eigen_iB*eigen_sW.asDiagonal()*eigen_K;
}

void CLaplacianInferenceMethod::update_chol()
{
	Map<VectorXd> eigen_W(W.vector, W.vlen);

	// create eigen representation of kernel matrix
	Map<MatrixXd> eigen_ktrtr(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	// create shogun and eigen representation of posterior cholesky
	m_L=SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	if (eigen_W.minCoeff() < 0)
	{
		// compute inverse of diagonal noise: iW = 1/W
		VectorXd eigen_iW = (VectorXd::Ones(W.vlen)).cwiseQuotient(eigen_W);

		FullPivLU<MatrixXd> lu(
			eigen_ktrtr*CMath::sq(m_scale)+MatrixXd(eigen_iW.asDiagonal()));

		// compute cholesky: L = -(K + iW)^-1
		eigen_L = -lu.inverse();
	}
	else
	{
		Map<VectorXd> eigen_sW(sW.vector, sW.vlen);

		// compute cholesky: L = chol(sW * sW' .* K + I)
		LLT<MatrixXd> L(
			(eigen_sW*eigen_sW.transpose()).cwiseProduct(eigen_ktrtr*CMath::sq(m_scale))+
			MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));

		eigen_L = L.matrixU();
	}
}

void CLaplacianInferenceMethod::update_alpha()
{
	float64_t Psi_Old = CMath::INFTY;
	float64_t Psi_New;
	float64_t Psi_Def;

	// get mean vector and create eigen representation of it
	SGVector<float64_t> mean=m_mean->get_mean_vector(m_feature_matrix);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);

	// create eigen representation of kernel matrix
	Map<MatrixXd> eigen_ktrtr(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	// create shogun and eigen representation of function vector
	m_approx_mean=SGVector<float64_t>(mean.vlen);
	Map<VectorXd> eigen_approx_mean(m_approx_mean, m_approx_mean.vlen);

	if (m_alpha.vlen!=m_labels->get_num_labels())
	{
		// set alpha a zero vector
		m_alpha=SGVector<float64_t>(m_labels->get_num_labels());
		m_alpha.zero();

		// f = mean, if length of alpha and length of y doesn't match
		eigen_approx_mean=eigen_mean;

		// compute W = -d2lp
		W=m_model->get_log_probability_derivative_f(m_labels, m_approx_mean, 2);
		W.scale(-1.0);

		Psi_New=-SGVector<float64_t>::sum(m_model->get_log_probability_f(
			m_labels, m_approx_mean));
	}
	else
	{
		Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

		// compute f = K * alpha + m
		eigen_approx_mean=eigen_ktrtr*CMath::sq(m_scale)*eigen_alpha+eigen_mean;

		// compute W = -d2lp
		W=m_model->get_log_probability_derivative_f(m_labels, m_approx_mean, 2);
		W.scale(-1.0);

		Psi_New=eigen_alpha.dot(eigen_approx_mean-eigen_mean)/2.0-
			SGVector<float64_t>::sum(m_model->get_log_probability_f(m_labels, m_approx_mean));

		Psi_Def=-SGVector<float64_t>::sum(m_model->get_log_probability_f(m_labels, mean));

		// if default is better, then use it
		if (Psi_Def < Psi_New)
		{
			m_alpha.zero();
			eigen_approx_mean=eigen_mean;
			Psi_New=-SGVector<float64_t>::sum(m_model->get_log_probability_f(
				m_labels, m_approx_mean));
		}
	}

	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	// get first derivative of log probability function
	dlp=m_model->get_log_probability_derivative_f(m_labels, m_approx_mean, 1);

	// create shogun and eigen representation of sW
	sW=SGVector<float64_t>(W.vlen);
	Map<VectorXd> eigen_sW(sW.vector, sW.vlen);

	index_t iter=0;

	while (Psi_Old-Psi_New>m_tolerance && iter<m_iter)
	{
		Map<VectorXd> eigen_W(W.vector, W.vlen);
		Map<VectorXd> eigen_dlp(dlp.vector, dlp.vlen);

		Psi_Old = Psi_New;
		iter++;

		if (eigen_W.minCoeff() < 0)
		{
			// Suggested by Vanhatalo et. al.,
			// Gaussian Process Regression with Student's t likelihood, NIPS 2009
			// Quoted from infLaplace.m
			float64_t df;

			if (m_model->get_model_type()==LT_STUDENTST)
			{
				CStudentsTLikelihood* lik=CStudentsTLikelihood::obtain_from_generic(m_model);
				df=lik->get_degrees_freedom();
				SG_UNREF(lik);
			}
			else
				df=1;

			eigen_W+=(2.0/df)*eigen_dlp.cwiseProduct(eigen_dlp);
		}

		// compute sW = sqrt(W)
		eigen_sW=eigen_W.cwiseSqrt();

		LLT<MatrixXd> L((eigen_sW*eigen_sW.transpose()).cwiseProduct(eigen_ktrtr*CMath::sq(m_scale))+
			MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));

		VectorXd b=eigen_W.cwiseProduct(eigen_approx_mean - eigen_mean)+eigen_dlp;

		VectorXd dalpha=b-eigen_sW.cwiseProduct(
			L.solve(eigen_sW.cwiseProduct(eigen_ktrtr*b*CMath::sq(m_scale))))-eigen_alpha;

		// perform Brent's optimization
		CPsiLine func;

		func.scale=m_scale;
		func.K=eigen_ktrtr;
		func.dalpha=dalpha;
		func.start_alpha=eigen_alpha;
		func.alpha=&eigen_alpha;
		func.dlp=&dlp;
		func.f=&m_approx_mean;
		func.m=&mean;
		func.W=&W;
		func.lik=m_model;
		func.lab=m_labels;

		float64_t x;
		Psi_New=local_min(0, m_opt_max, m_opt_tolerance, func, x);
	}

	// compute f = K * alpha + m
	eigen_approx_mean=eigen_ktrtr*CMath::sq(m_scale)*eigen_alpha+eigen_mean;

	// get log probability derivatives
	dlp=m_model->get_log_probability_derivative_f(m_labels, m_approx_mean, 1);
	d2lp=m_model->get_log_probability_derivative_f(m_labels, m_approx_mean, 2);
	d3lp=m_model->get_log_probability_derivative_f(m_labels, m_approx_mean, 3);

	// W = -d2lp
	W = d2lp.clone();
	W.scale(-1.0);

	// compute sW
	Map<VectorXd> eigen_W(W.vector, W.vlen);

	if (eigen_W.minCoeff() > 0)
		eigen_sW=eigen_W.cwiseSqrt();
	else
		eigen_sW.setZero();
}
}

#endif // HAVE_EIGEN3
