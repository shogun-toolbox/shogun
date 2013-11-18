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
 */

#include <shogun/machine/gp/FITCInferenceMethod.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CFITCInferenceMethod::CFITCInferenceMethod() : CInferenceMethod()
{
	init();
}

CFITCInferenceMethod::CFITCInferenceMethod(CKernel* kern, CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod, CFeatures* lat)
		: CInferenceMethod(kern, feat, m, lab, mod)
{
	init();
	set_latent_features(lat);
}

void CFITCInferenceMethod::init()
{
	SG_ADD((CSGObject**)&m_latent_features, "latent_features", "Latent features",
			MS_NOT_AVAILABLE);

	m_latent_features=NULL;
	m_ind_noise=1e-10;
}

CFITCInferenceMethod::~CFITCInferenceMethod()
{
	SG_UNREF(m_latent_features);
}

CFITCInferenceMethod* CFITCInferenceMethod::obtain_from_generic(
		CInferenceMethod* inference)
{
	ASSERT(inference!=NULL);

	if (inference->get_inference_type()!=INF_FITC)
		SG_SERROR("Provided inference is not of type CFITCInferenceMethod!\n")

	SG_REF(inference);
	return (CFITCInferenceMethod*)inference;
}

void CFITCInferenceMethod::update()
{
	CInferenceMethod::update();
	update_chol();
	update_alpha();
	update_deriv();
}

void CFITCInferenceMethod::check_members() const
{
	CInferenceMethod::check_members();

	REQUIRE(m_model->get_model_type()==LT_GAUSSIAN,
			"FITC inference method can only use Gaussian likelihood function\n")
	REQUIRE(m_labels->get_label_type()==LT_REGRESSION, "Labels must be type "
			"of CRegressionLabels\n")
	REQUIRE(m_latent_features, "Latent features should not be NULL\n")
	REQUIRE(m_latent_features->get_num_vectors(),
			"Number of latent features must be greater than zero\n")
}

SGVector<float64_t> CFITCInferenceMethod::get_diagonal_vector()
{
	if (update_parameter_hash())
		update();

	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	// compute diagonal vector: sW=1/sigma
	SGVector<float64_t> result(m_features->get_num_vectors());
	result.fill_vector(result.vector, m_features->get_num_vectors(), 1.0/sigma);

	return result;
}

float64_t CFITCInferenceMethod::get_negative_log_marginal_likelihood()
{
	if (update_parameter_hash())
		update();

	// create eigen representations of chol_utr, dg, r, be
    Map<MatrixXd> eigen_chol_utr(m_chol_utr.matrix, m_chol_utr.num_rows,
			m_chol_utr.num_cols);
	Map<VectorXd> eigen_dg(m_dg.vector, m_dg.vlen);
	Map<VectorXd> eigen_r(m_r.vector, m_r.vlen);
	Map<VectorXd> eigen_be(m_be.vector, m_be.vlen);

	// compute negative log marginal likelihood:
	// nlZ=sum(log(diag(utr)))+(sum(log(dg))+r'*r-be'*be+n*log(2*pi))/2
	float64_t result=eigen_chol_utr.diagonal().array().log().sum()+
		(eigen_dg.array().log().sum()+eigen_r.dot(eigen_r)-eigen_be.dot(eigen_be)+
		 m_ktrtr.num_rows*CMath::log(2*CMath::PI))/2.0;

	return result;
}

SGVector<float64_t> CFITCInferenceMethod::get_alpha()
{
	if (update_parameter_hash())
		update();

	SGVector<float64_t> result(m_alpha);
	return result;
}

SGMatrix<float64_t> CFITCInferenceMethod::get_cholesky()
{
	if (update_parameter_hash())
		update();

	SGMatrix<float64_t> result(m_L);
	return result;
}

SGVector<float64_t> CFITCInferenceMethod::get_posterior_mean()
{
	SG_NOTIMPLEMENTED
	return SGVector<float64_t>();
}

SGMatrix<float64_t> CFITCInferenceMethod::get_posterior_covariance()
{
	SG_NOTIMPLEMENTED
	return SGMatrix<float64_t>();
}

void CFITCInferenceMethod::update_train_kernel()
{
	CInferenceMethod::update_train_kernel();

	// create kernel matrix for latent features
	m_kernel->cleanup();
	m_kernel->init(m_latent_features, m_latent_features);
	m_kuu=m_kernel->get_kernel_matrix();

	// create kernel matrix for latent and training features
	m_kernel->cleanup();
	m_kernel->init(m_latent_features, m_features);
	m_ktru=m_kernel->get_kernel_matrix();
}

void CFITCInferenceMethod::update_chol()
{
	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	// eigen3 representation of covariance matrix of latent features (m_kuu)
	// and training features (m_ktru)
	Map<MatrixXd> eigen_kuu(m_kuu.matrix, m_kuu.num_rows, m_kuu.num_cols);
	Map<MatrixXd> eigen_ktru(m_ktru.matrix, m_ktru.num_rows, m_ktru.num_cols);
	Map<MatrixXd> eigen_ktrtr(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	// solve Luu' * Luu = Kuu + m_ind_noise * I
	LLT<MatrixXd> Luu(eigen_kuu*CMath::sq(m_scale)+m_ind_noise*MatrixXd::Identity(
		m_kuu.num_rows, m_kuu.num_cols));

	// create shogun and eigen3 representation of cholesky of covariance of
    // latent features Luu (m_chol_uu and eigen_chol_uu)
	m_chol_uu=SGMatrix<float64_t>(Luu.rows(), Luu.cols());
	Map<MatrixXd> eigen_chol_uu(m_chol_uu.matrix, m_chol_uu.num_rows,
		m_chol_uu.num_cols);
	eigen_chol_uu=Luu.matrixU();

	// solve Luu' * V = Ktru
	MatrixXd V=eigen_chol_uu.triangularView<Upper>().adjoint().solve(eigen_ktru*
			CMath::sq(m_scale));

	// create shogun and eigen3 representation of
	// dg = diag(K) + sn2 - diag(Q)
	m_dg=SGVector<float64_t>(m_ktrtr.num_cols);
	Map<VectorXd> eigen_dg(m_dg.vector, m_dg.vlen);

	eigen_dg=eigen_ktrtr.diagonal()*CMath::sq(m_scale)+CMath::sq(sigma)*
		VectorXd::Ones(m_dg.vlen)-(V.cwiseProduct(V)).colwise().sum().adjoint();

	// solve Lu' * Lu = V * diag(1/dg) * V' + I
	LLT<MatrixXd> Lu(V*((VectorXd::Ones(m_dg.vlen)).cwiseQuotient(eigen_dg)).asDiagonal()*
			V.adjoint()+MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols));

	// create shogun and eigen3 representation of cholesky of covariance of
    // training features Luu (m_chol_utr and eigen_chol_utr)
	m_chol_utr=SGMatrix<float64_t>(Lu.rows(), Lu.cols());
	Map<MatrixXd> eigen_chol_utr(m_chol_utr.matrix, m_chol_utr.num_rows,
		m_chol_utr.num_cols);
	eigen_chol_utr=Lu.matrixU();

	// create eigen representation of labels and mean vectors
	SGVector<float64_t> y=((CRegressionLabels*) m_labels)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);
	SGVector<float64_t> m=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_m(m.vector, m.vlen);

	// compute sgrt_dg = sqrt(dg)
	VectorXd sqrt_dg=eigen_dg.array().sqrt();

	// create shogun and eigen3 representation of labels adjusted for
	// noise and means (m_r)
	m_r=SGVector<float64_t>(y.vlen);
	Map<VectorXd> eigen_r(m_r.vector, m_r.vlen);
	eigen_r=(eigen_y-eigen_m).cwiseQuotient(sqrt_dg);

	// compute be
	m_be=SGVector<float64_t>(m_chol_utr.num_cols);
	Map<VectorXd> eigen_be(m_be.vector, m_be.vlen);
	eigen_be=eigen_chol_utr.triangularView<Upper>().adjoint().solve(
		V*eigen_r.cwiseQuotient(sqrt_dg));

	// compute iKuu
	MatrixXd iKuu=Luu.solve(MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols));

	// create shogun and eigen3 representation of posterior cholesky
	MatrixXd eigen_prod=eigen_chol_utr*eigen_chol_uu;
	m_L=SGMatrix<float64_t>(m_kuu.num_rows, m_kuu.num_cols);
	Map<MatrixXd> eigen_chol(m_L.matrix, m_L.num_rows, m_L.num_cols);

	eigen_chol=eigen_prod.triangularView<Upper>().adjoint().solve(
		MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols));
	eigen_chol=eigen_prod.triangularView<Upper>().solve(eigen_chol)-iKuu;
}

void CFITCInferenceMethod::update_alpha()
{
    Map<MatrixXd> eigen_chol_uu(m_chol_uu.matrix, m_chol_uu.num_rows,
		m_chol_uu.num_cols);
	Map<MatrixXd> eigen_chol_utr(m_chol_utr.matrix, m_chol_utr.num_rows,
		m_chol_utr.num_cols);
	Map<VectorXd> eigen_be(m_be.vector, m_be.vlen);

	// create shogun and eigen representations of alpha
	// and solve Luu * Lu * alpha = be
	m_alpha=SGVector<float64_t>(m_chol_uu.num_rows);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	eigen_alpha=eigen_chol_utr.triangularView<Upper>().solve(eigen_be);
	eigen_alpha=eigen_chol_uu.triangularView<Upper>().solve(eigen_alpha);
}

void CFITCInferenceMethod::update_deriv()
{
	// create eigen representation of Ktru, Lu, Luu, dg, be
	Map<MatrixXd> eigen_Ktru(m_ktru.matrix, m_ktru.num_rows, m_ktru.num_cols);
	Map<MatrixXd> eigen_Lu(m_chol_utr.matrix, m_chol_utr.num_rows,
			m_chol_utr.num_cols);
	Map<MatrixXd> eigen_Luu(m_chol_uu.matrix, m_chol_uu.num_rows,
			m_chol_uu.num_cols);
	Map<VectorXd> eigen_dg(m_dg.vector, m_dg.vlen);
	Map<VectorXd> eigen_be(m_be.vector, m_be.vlen);

	// get and create eigen representation of labels
	SGVector<float64_t> y=((CRegressionLabels*) m_labels)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// get and create eigen representation of mean vector
	SGVector<float64_t> m=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_m(m.vector, m.vlen);

	// compute V=inv(Luu')*Ku
	MatrixXd V=eigen_Luu.triangularView<Upper>().adjoint().solve(eigen_Ktru*
			CMath::sq(m_scale));

	// create shogun and eigen representation of al
	m_al=SGVector<float64_t>(m.vlen);
	Map<VectorXd> eigen_al(m_al.vector, m_al.vlen);

	// compute al=(Kt+sn2*eye(n))\y
	eigen_al=((eigen_y-eigen_m)-(V.adjoint()*
		eigen_Lu.triangularView<Upper>().solve(eigen_be))).cwiseQuotient(eigen_dg);

	// compute inv(Kuu+snu2*I)=iKuu
	MatrixXd iKuu=eigen_Luu.triangularView<Upper>().adjoint().solve(
			MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols));
	iKuu=eigen_Luu.triangularView<Upper>().solve(iKuu);

	// create shogun and eigen representation of B
	m_B=SGMatrix<float64_t>(iKuu.rows(), eigen_Ktru.cols());
	Map<MatrixXd> eigen_B(m_B.matrix, m_B.num_rows, m_B.num_cols);

	eigen_B=iKuu*eigen_Ktru*CMath::sq(m_scale);

	// create shogun and eigen representation of w
	m_w=SGVector<float64_t>(m_B.num_rows);
	Map<VectorXd> eigen_w(m_w.vector, m_w.vlen);

	eigen_w=eigen_B*eigen_al;

	// create shogun and eigen representation of W
	m_W=SGMatrix<float64_t>(m_chol_utr.num_cols, m_dg.vlen);
	Map<MatrixXd> eigen_W(m_W.matrix, m_W.num_rows, m_W.num_cols);

	// compute W=Lu'\(V./repmat(g_sn2',nu,1))
	eigen_W=eigen_Lu.triangularView<Upper>().adjoint().solve(V*VectorXd::Ones(
		m_dg.vlen).cwiseQuotient(eigen_dg).asDiagonal());
}

SGVector<float64_t> CFITCInferenceMethod::get_derivative_wrt_inference_method(
		const TParameter* param)
{
	REQUIRE(!strcmp(param->m_name, "scale"), "Can't compute derivative of "
			"the nagative log marginal likelihood wrt %s.%s parameter\n",
			get_name(), param->m_name)

	// create eigen representation of dg, al, B, W, w
	Map<VectorXd> eigen_dg(m_dg.vector, m_dg.vlen);
	Map<VectorXd> eigen_al(m_al.vector, m_al.vlen);
	Map<VectorXd> eigen_w(m_w.vector, m_w.vlen);
	Map<MatrixXd> eigen_B(m_B.matrix, m_B.num_rows, m_B.num_cols);
	Map<MatrixXd> eigen_W(m_W.matrix, m_W.num_rows, m_W.num_cols);

	// clone kernel matrices
	SGVector<float64_t> deriv_trtr=m_ktrtr.get_diagonal_vector();
	SGMatrix<float64_t> deriv_uu=m_kuu.clone();
	SGMatrix<float64_t> deriv_tru=m_ktru.clone();

	// create eigen representation of kernel matrices
	Map<VectorXd> ddiagKi(deriv_trtr.vector, deriv_trtr.vlen);
	Map<MatrixXd> dKuui(deriv_uu.matrix, deriv_uu.num_rows, deriv_uu.num_cols);
	Map<MatrixXd> dKui(deriv_tru.matrix, deriv_tru.num_rows, deriv_tru.num_cols);

	// compute derivatives wrt scale for each kernel matrix
	ddiagKi*=m_scale*2.0;
	dKuui*=m_scale*2.0;
	dKui*=m_scale*2.0;

	// compute R=2*dKui-dKuui*B
	MatrixXd R=2*dKui-dKuui*eigen_B;

	// compute v=ddiagKi-sum(R.*B, 1)'
	VectorXd v=ddiagKi-R.cwiseProduct(eigen_B).colwise().sum().adjoint();

	SGVector<float64_t> result(1);

	// compute dnlZ=(ddiagKi'*(1./g_sn2)+w'*(dKuui*w-2*(dKui*al))-al'*(v.*al)-
	// sum(W.*W,1)*v- sum(sum((R*W').*(B*W'))))/2;
	result[0]=(ddiagKi.dot(VectorXd::Ones(m_dg.vlen).cwiseQuotient(eigen_dg))+
			eigen_w.dot(dKuui*eigen_w-2*(dKui*eigen_al))-
			eigen_al.dot(v.cwiseProduct(eigen_al))-
			eigen_W.cwiseProduct(eigen_W).colwise().sum().dot(v)-
			(R*eigen_W.adjoint()).cwiseProduct(eigen_B*eigen_W.adjoint()).sum())/2.0;

	return result;
}

SGVector<float64_t> CFITCInferenceMethod::get_derivative_wrt_likelihood_model(
		const TParameter* param)
{
	REQUIRE(!strcmp(param->m_name, "sigma"), "Can't compute derivative of "
			"the nagative log marginal likelihood wrt %s.%s parameter\n",
			m_model->get_name(), param->m_name)

	// create eigen representation of dg, al, w, W and B
	Map<VectorXd> eigen_dg(m_dg.vector, m_dg.vlen);
	Map<VectorXd> eigen_al(m_al.vector, m_al.vlen);
	Map<VectorXd> eigen_w(m_w.vector, m_w.vlen);
	Map<MatrixXd> eigen_W(m_W.matrix, m_W.num_rows, m_W.num_cols);
	Map<MatrixXd> eigen_B(m_B.matrix, m_B.num_rows, m_B.num_cols);

	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	SGVector<float64_t> result(1);

	result[0]=CMath::sq(sigma)*(VectorXd::Ones(m_dg.vlen).cwiseQuotient(
		eigen_dg).sum()-eigen_W.cwiseProduct(eigen_W).sum()-eigen_al.dot(eigen_al));

	float64_t dKuui=2.0*m_ind_noise;
	MatrixXd R=-dKuui*eigen_B;
	VectorXd v=-R.cwiseProduct(eigen_B).colwise().sum().adjoint();

	result[0]=result[0]+((eigen_w.dot(dKuui*eigen_w))-eigen_al.dot(
		v.cwiseProduct(eigen_al))-eigen_W.cwiseProduct(eigen_W).colwise().sum().dot(v)-
		(R*eigen_W.adjoint()).cwiseProduct(eigen_B*eigen_W.adjoint()).sum())/2.0;

	return result;
}

SGVector<float64_t> CFITCInferenceMethod::get_derivative_wrt_kernel(
		const TParameter* param)
{
	// create eigen representation of dg, al, w, W, B
	Map<VectorXd> eigen_dg(m_dg.vector, m_dg.vlen);
	Map<VectorXd> eigen_al(m_al.vector, m_al.vlen);
	Map<VectorXd> eigen_w(m_w.vector, m_w.vlen);
	Map<MatrixXd> eigen_W(m_W.matrix, m_W.num_rows, m_W.num_cols);
	Map<MatrixXd> eigen_B(m_B.matrix, m_B.num_rows, m_B.num_cols);

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
		SGVector<float64_t> deriv_trtr;
		SGMatrix<float64_t> deriv_uu;
		SGMatrix<float64_t> deriv_tru;

		if (result.vlen==1)
		{
			m_kernel->init(m_features, m_features);
			deriv_trtr=m_kernel->get_parameter_gradient(param).get_diagonal_vector();

			m_kernel->init(m_latent_features, m_latent_features);
			deriv_uu=m_kernel->get_parameter_gradient(param);

			m_kernel->init(m_latent_features, m_features);
			deriv_tru=m_kernel->get_parameter_gradient(param);
		}
		else
		{
			m_kernel->init(m_features, m_features);
			deriv_trtr=m_kernel->get_parameter_gradient(param, i).get_diagonal_vector();

			m_kernel->init(m_latent_features, m_latent_features);
			deriv_uu=m_kernel->get_parameter_gradient(param, i);

			m_kernel->init(m_latent_features, m_features);
			deriv_tru=m_kernel->get_parameter_gradient(param, i);
		}

		m_kernel->cleanup();

		// create eigen representation of derivatives
		Map<VectorXd> ddiagKi(deriv_trtr.vector, deriv_trtr.vlen);
		Map<MatrixXd> dKuui(deriv_uu.matrix, deriv_uu.num_rows,
				deriv_uu.num_cols);
		Map<MatrixXd> dKui(deriv_tru.matrix, deriv_tru.num_rows,
				deriv_tru.num_cols);

		ddiagKi*=CMath::sq(m_scale);
		dKuui*=CMath::sq(m_scale);
		dKui*=CMath::sq(m_scale);

		// compute R=2*dKui-dKuui*B
		MatrixXd R=2*dKui-dKuui*eigen_B;

		// compute v=ddiagKi-sum(R.*B, 1)'
		VectorXd v=ddiagKi-R.cwiseProduct(eigen_B).colwise().sum().adjoint();

		// compute dnlZ=(ddiagKi'*(1./g_sn2)+w'*(dKuui*w-2*(dKui*al))-al'*
		// (v.*al)-sum(W.*W,1)*v- sum(sum((R*W').*(B*W'))))/2;
		result[i]=(ddiagKi.dot(VectorXd::Ones(m_dg.vlen).cwiseQuotient(eigen_dg))+
				eigen_w.dot(dKuui*eigen_w-2*(dKui*eigen_al))-
				eigen_al.dot(v.cwiseProduct(eigen_al))-
				eigen_W.cwiseProduct(eigen_W).colwise().sum().dot(v)-
				(R*eigen_W.adjoint()).cwiseProduct(eigen_B*eigen_W.adjoint()).sum())/2.0;
	}

	return result;
}

SGVector<float64_t> CFITCInferenceMethod::get_derivative_wrt_mean(
		const TParameter* param)
{
	// create eigen representation of al vector
	Map<VectorXd> eigen_al(m_al.vector, m_al.vlen);

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

		if (result.vlen==1)
			dmu=m_mean->get_parameter_derivative(m_features, param);
		else
			dmu=m_mean->get_parameter_derivative(m_features, param, i);

		Map<VectorXd> eigen_dmu(dmu.vector, dmu.vlen);

		// compute dnlZ=-dm'*al
		result[i]=-eigen_dmu.dot(eigen_al);
	}

	return result;
}

#endif /* HAVE_EIGEN3 */
