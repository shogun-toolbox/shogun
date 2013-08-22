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
#include <shogun/features/CombinedFeatures.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CFITCInferenceMethod::CFITCInferenceMethod() : CInferenceMethod()
{
	init();
}

CFITCInferenceMethod::CFITCInferenceMethod(CKernel* kern, CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod, CFeatures* lat,
		float64_t scale) : CInferenceMethod(kern, feat, m, lab, mod, scale)
{
	init();
	set_latent_features(lat);
}

void CFITCInferenceMethod::init()
{
	SG_ADD((CSGObject**)&m_latent_features, "latent_features", "latent Features",
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

	CFeatures* feat=m_latent_features;

	if (m_latent_features->get_feature_class()==C_COMBINED)
		feat=((CCombinedFeatures*)m_latent_features)->get_first_feature_obj();
	else
		SG_REF(m_latent_features);

	REQUIRE(feat->has_property(FP_DOT),
			"Latent features must be type of CFeatures\n")
	REQUIRE(feat->get_feature_class()==C_DENSE,
			"Latent features must be dense\n")
	REQUIRE(feat->get_feature_type()==F_DREAL, "Latent features must be real\n")

	SG_UNREF(feat);
}

CMap<TParameter*, SGVector<float64_t> > CFITCInferenceMethod::
get_marginal_likelihood_derivatives(CMap<TParameter*, CSGObject*>& para_dict)
{
	if (update_parameter_hash())
		update();

	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	Map<MatrixXd> eigen_ktru(m_ktru.matrix, m_ktru.num_rows, m_ktru.num_cols);

	MatrixXd W = eigen_ktru;

	Map<VectorXd> eigen_dg(m_dg.vector, m_dg.vlen);

	for (index_t j = 0; j < eigen_ktru.rows(); j++)
	{
		for (index_t i = 0; i < eigen_ktru.cols(); i++)
			W(i,j) = eigen_ktru(i,j) / sqrt(eigen_dg[j]);
	}

	Map<MatrixXd> eigen_uu(m_kuu.matrix, m_kuu.num_rows, m_kuu.num_cols);
	LLT<MatrixXd> CholW(eigen_uu + W*W.transpose() +
			m_ind_noise*MatrixXd::Identity(eigen_uu.rows(), eigen_uu.cols()));
	W = CholW.matrixL();


	W = W.colPivHouseholderQr().solve(eigen_ktru);

	SGVector<float64_t> y=((CRegressionLabels*) m_labels)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);
	SGVector<float64_t> m=m_mean->get_mean_vector(m_feat);
	Map<VectorXd> eigen_m(m.vector, m.vlen);

	VectorXd al=W*(eigen_y-eigen_m).cwiseQuotient(eigen_dg);

	al = W.transpose()*al;

	al=(eigen_y-eigen_m)-al;

	al = al.cwiseQuotient(eigen_dg);

	MatrixXd iKuu = eigen_uu.selfadjointView<Eigen::Upper>().llt()
					.solve(MatrixXd::Identity(eigen_uu.rows(), eigen_uu.cols()));

	MatrixXd B = iKuu*eigen_ktru;

	MatrixXd Wdg = W;

	for (index_t j = 0; j < eigen_ktru.rows(); j++)
	{
		for (index_t i = 0; i < eigen_ktru.cols(); i++)
			Wdg(i,j) = Wdg(i,j) / eigen_dg[j];
	}

	VectorXd w = B*al;

	VectorXd sum(1);
	sum[0] = 0;

	m_kernel->build_parameter_dictionary(para_dict);
	m_mean->build_parameter_dictionary(para_dict);

	//This will be the vector we return
	CMap<TParameter*, SGVector<float64_t> > gradient(
			3+para_dict.get_num_elements(),
			3+para_dict.get_num_elements());

	for (index_t i = 0; i < para_dict.get_num_elements(); i++)
	{
		shogun::CMapNode<TParameter*, CSGObject*>* node =
				para_dict.get_node_ptr(i);

		TParameter* param = node->key;
		CSGObject* obj = node->data;

		index_t length = 1;

		if ((param->m_datatype.m_ctype== CT_VECTOR ||
				param->m_datatype.m_ctype == CT_SGVECTOR) &&
				param->m_datatype.m_length_y != NULL)
			length = *(param->m_datatype.m_length_y);

		SGVector<float64_t> variables(length);

		bool deriv_found = false;

		for (index_t g = 0; g < length; g++)
		{

			SGMatrix<float64_t> deriv;
			SGMatrix<float64_t> derivtru;
			SGMatrix<float64_t> derivuu;
			SGVector<float64_t> mean_derivatives;
			VectorXd mean_dev_temp;

			if (param->m_datatype.m_ctype == CT_VECTOR ||
					param->m_datatype.m_ctype == CT_SGVECTOR)
			{
				m_kernel->init(m_features, m_features);
				deriv = m_kernel->get_parameter_gradient(param, obj);

				m_kernel->init(m_latent_features, m_features);
				derivtru = m_kernel->get_parameter_gradient(param, obj);

				m_kernel->init(m_latent_features, m_latent_features);
				derivuu = m_kernel->get_parameter_gradient(param, obj);

				m_kernel->remove_lhs_and_rhs();

				mean_derivatives = m_mean->get_parameter_derivative(
						param, obj, m_feat, g);

				for (index_t d = 0; d < mean_derivatives.vlen; d++)
					mean_dev_temp[d] = mean_derivatives[d];
			}

			else
			{
				mean_derivatives = m_mean->get_parameter_derivative(
						param, obj, m_feat);

				for (index_t d = 0; d < mean_derivatives.vlen; d++)
					mean_dev_temp[d] = mean_derivatives[d];

				m_kernel->init(m_features, m_features);
				deriv = m_kernel->get_parameter_gradient(param, obj);

				m_kernel->init(m_latent_features, m_features);
				derivtru = m_kernel->get_parameter_gradient(param, obj);

				m_kernel->init(m_latent_features, m_latent_features);
				derivuu = m_kernel->get_parameter_gradient(param, obj);

				m_kernel->remove_lhs_and_rhs();
			}

			sum[0] = 0;


			if (deriv.num_cols*deriv.num_rows > 0)
			{
				MatrixXd ddiagKi(deriv.num_cols, deriv.num_rows);
				MatrixXd dKuui(derivuu.num_cols, derivuu.num_rows);
				MatrixXd dKui(derivtru.num_cols, derivtru.num_rows);

				for (index_t d = 0; d < deriv.num_rows; d++)
				{
					for (index_t s = 0; s < deriv.num_cols; s++)
						ddiagKi(d,s) = deriv(d,s)*m_scale*m_scale;
				}

				for (index_t d = 0; d < derivuu.num_rows; d++)
				{
					for (index_t s = 0; s < derivuu.num_cols; s++)
						dKuui(d,s) = derivuu(d,s)*m_scale*m_scale;
				}

				for (index_t d = 0; d < derivtru.num_rows; d++)
				{
					for (index_t s = 0; s < derivtru.num_cols; s++)
						dKui(d,s) = derivtru(d,s)*m_scale*m_scale;
				}

				MatrixXd R = 2*dKui-dKuui*B;
				MatrixXd v = ddiagKi;
				MatrixXd temp = R.cwiseProduct(B);

				for (index_t d = 0; d < ddiagKi.rows(); d++)
					v(d,d) = v(d,d) - temp.col(d).sum();

				sum = sum + ddiagKi.diagonal().transpose()*
						VectorXd::Ones(eigen_dg.rows()).cwiseQuotient(eigen_dg);

				sum = sum + w.transpose()*(dKuui*w-2*(dKui*al));

				sum = sum - al.transpose()*(v.diagonal().cwiseProduct(al));

				MatrixXd Wdg_temp = Wdg.cwiseProduct(Wdg);

				VectorXd Wdg_sum(Wdg.rows());

				for (index_t d = 0; d < Wdg.rows(); d++)
					Wdg_sum[d] = Wdg_temp.col(d).sum();

				sum = sum - v.diagonal().transpose()*Wdg_sum;

				Wdg_temp = (R*Wdg.transpose()).cwiseProduct(B*Wdg.transpose());

				sum[0] = sum[0] - Wdg_temp.sum();

				sum /= 2.0;

				variables[g] = sum[0];
				deriv_found = true;
			}

			else if (mean_derivatives.vlen > 0)
			{
				sum = mean_dev_temp*al;
				variables[g] = sum[0];
				deriv_found = true;
			}


		}

		if (deriv_found)
			gradient.add(param, variables);

	}

	//Here we take the kernel scale derivative.
	{
		TParameter* param;
		index_t index = get_modsel_param_index("scale");
		param = m_model_selection_parameters->get_parameter(index);

		SGVector<float64_t> variables(1);

		SGMatrix<float64_t> deriv;
		SGMatrix<float64_t> derivtru;
		SGMatrix<float64_t> derivuu;

		m_kernel->init(m_features, m_features);
		deriv = m_kernel->get_kernel_matrix();

		m_kernel->init(m_latent_features, m_features);
		derivtru = m_kernel->get_kernel_matrix();

		m_kernel->init(m_latent_features, m_latent_features);
		derivuu = m_kernel->get_kernel_matrix();

		m_kernel->remove_lhs_and_rhs();

		MatrixXd ddiagKi(deriv.num_cols, deriv.num_rows);
		MatrixXd dKuui(derivuu.num_cols, derivuu.num_rows);
		MatrixXd dKui(derivtru.num_cols, derivtru.num_rows);

		for (index_t d = 0; d < deriv.num_rows; d++)
		{
			for (index_t s = 0; s < deriv.num_cols; s++)
				ddiagKi(d,s) = deriv(d,s)*m_scale*2.0;
		}

		for (index_t d = 0; d < derivuu.num_rows; d++)
		{
			for (index_t s = 0; s < derivuu.num_cols; s++)
				dKuui(d,s) = derivuu(d,s)*m_scale*2.0;
		}

		for (index_t d = 0; d < derivtru.num_rows; d++)
		{
			for (index_t s = 0; s < derivtru.num_cols; s++)
				dKui(d,s) = derivtru(d,s)*m_scale*2.0;
		}

		MatrixXd R = 2*dKui-dKuui*B;
		MatrixXd v = ddiagKi;
		MatrixXd temp = R.cwiseProduct(B);

		for (index_t d = 0; d < ddiagKi.rows(); d++)
			v(d,d) = v(d,d) - temp.col(d).sum();

		sum = sum + ddiagKi.diagonal().transpose()*

				VectorXd::Ones(eigen_dg.rows()).cwiseQuotient(eigen_dg);

		sum = sum + w.transpose()*(dKuui*w-2*(dKui*al));

		sum = sum - al.transpose()*(v.diagonal().cwiseProduct(al));

		MatrixXd Wdg_temp = Wdg.cwiseProduct(Wdg);

		VectorXd Wdg_sum(Wdg.rows());

		for (index_t d = 0; d < Wdg.rows(); d++)
			Wdg_sum[d] = Wdg_temp.col(d).sum();

		sum = sum - v.diagonal().transpose()*Wdg_sum;

		Wdg_temp = (R*Wdg.transpose()).cwiseProduct(B*Wdg.transpose());

		sum[0] = sum[0] - Wdg_temp.sum();

		sum /= 2.0;

		variables[0] = sum[0];

		gradient.add(param, variables);
		para_dict.add(param, this);

	}

	TParameter* param;
	index_t index;

	index = m_model->get_modsel_param_index("sigma");
	param = m_model->m_model_selection_parameters->get_parameter(index);

	sum[0] = 0;

	MatrixXd W_temp = W.cwiseProduct(W);
	VectorXd W_sum(W_temp.rows());

	for (index_t d = 0; d < W_sum.rows(); d++)
		W_sum[d] = W_temp.col(d).sum();

	W_sum = W_sum.cwiseQuotient(eigen_dg.cwiseProduct(eigen_dg));

	sum[0] = W_sum.sum();

	sum = sum + al.transpose()*al;

	sum[0] = VectorXd::Ones(eigen_dg.rows()).cwiseQuotient(eigen_dg).sum() - sum[0];

	sum = sum*sigma*sigma;
	float64_t dKuui = 2.0*m_ind_noise;

	MatrixXd R = -dKuui*B;

	MatrixXd temp = R.cwiseProduct(B);
	VectorXd v(temp.rows());

	for (index_t d = 0; d < temp.rows(); d++)
		v[d] = temp.col(d).sum();

	sum = sum + (w.transpose()*dKuui*w)/2.0;

	sum = sum - al.transpose()*(v.cwiseProduct(al))/2.0;

	MatrixXd Wdg_temp = Wdg.cwiseProduct(Wdg);
	VectorXd Wdg_sum(Wdg.rows());

	for (index_t d = 0; d < Wdg.rows(); d++)
		Wdg_sum[d] = Wdg_temp.col(d).sum();

	sum = sum - v.transpose()*Wdg_sum/2.0;


	Wdg_temp = (R*Wdg.transpose()).cwiseProduct(B*Wdg.transpose());

	sum[0] = sum[0] - Wdg_temp.sum()/2.0;

	SGVector<float64_t> vsigma(1);

	vsigma[0] = sum[0];
	gradient.add(param, vsigma);
	para_dict.add(param, m_model);

	return gradient;

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

float64_t CFITCInferenceMethod::get_negative_marginal_likelihood()
{
	if (update_parameter_hash())
		update();

	// create eigen representations of chol_utr, dg, r, be
    Map<MatrixXd> eigen_chol_utr(m_chol_utr.matrix, m_chol_utr.num_rows, m_chol_utr.num_cols);
	Map<VectorXd> eigen_dg(m_dg.vector, m_dg.vlen);
	Map<VectorXd> eigen_r(m_r.vector, m_r.vlen);
	Map<VectorXd> eigen_be(m_be.vector, m_be.vlen);

	// compute negative log marginal likelihood:
	// nlZ=sum(log(diag(utr)))+(sum(log(dg))+r'*r-be'*be+n*log(2*pi))/2
	float64_t result=eigen_chol_utr.diagonal().array().log().sum()+
		(eigen_dg.array().log().sum()+eigen_r.dot(eigen_r)-eigen_be.dot(eigen_be)+
		 m_chol_utr.num_rows*CMath::log(2*CMath::PI))/2.0;

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

void CFITCInferenceMethod::update_train_kernel()
{
	CInferenceMethod::update_train_kernel();

	// create kernel matrix for latent features
	m_kernel->cleanup();
	m_kernel->init(m_latent_features, m_latent_features);
	m_kuu=m_kernel->get_kernel_matrix();
	Map<MatrixXd> eigen_kuu(m_kuu.matrix, m_kuu.num_rows, m_kuu.num_cols);
	eigen_kuu*=CMath::sq(m_scale);

	// create kernel matrix for latent and training features
	m_kernel->cleanup();
	m_kernel->init(m_latent_features, m_features);
	m_ktru=m_kernel->get_kernel_matrix();
	Map<MatrixXd> eigen_ktru(m_ktru.matrix, m_ktru.num_rows, m_ktru.num_cols);
	eigen_ktru*=CMath::sq(m_scale);
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

	// solve Luu' * Luu = Kuu + m_ind_noise * I
	LLT<MatrixXd> Luu(eigen_kuu+m_ind_noise*MatrixXd::Identity(
		m_kuu.num_rows, m_kuu.num_cols));

	// create shogun and eigen3 representation of cholesky of covariance of
    // latent features Luu (m_chol_uu and eigen_chol_uu)
	m_chol_uu=SGMatrix<float64_t>(Luu.rows(), Luu.cols());
	Map<MatrixXd> eigen_chol_uu(m_chol_uu.matrix, m_chol_uu.num_rows,
		m_chol_uu.num_cols);
	eigen_chol_uu=Luu.matrixU();

	// solve Luu' * V = Ktru, and calculate sV = V.^2
	MatrixXd V = eigen_chol_uu.triangularView<Upper>().adjoint().solve(eigen_ktru);
	MatrixXd sV = V.cwiseProduct(V);

	// create shogun and eigen3 representation of
	// dg = diag(K) + sn2 - diag(Q), and also compute idg = 1/dg
	m_dg = SGVector<float64_t>(m_ktrtr.num_cols);
	Map<VectorXd> eigen_dg(m_dg.vector, m_dg.vlen);
	VectorXd eigen_idg(m_dg.vlen);

	for (index_t i = 0; i < m_ktrtr.num_cols; i++)
	{
		eigen_dg[i]=m_ktrtr(i,i)*m_scale*m_scale+sigma*sigma-sV.col(i).sum();
		eigen_idg[i] = 1.0 / eigen_dg[i];
	}

	// solve Lu' * Lu = V * diag(idg) * V' + I
	LLT<MatrixXd> Lu(V*eigen_idg.asDiagonal()*V.transpose()+
			MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols));

	// create shogun and eigen3 representation of cholesky of covariance of
    // training features Luu (m_chol_utr and eigen_chol_utr)
	m_chol_utr = SGMatrix<float64_t>(Lu.rows(),	Lu.cols());
	Map<MatrixXd> eigen_chol_utr(m_chol_utr.matrix, m_chol_utr.num_rows,
		m_chol_utr.num_rows);
	eigen_chol_utr = Lu.matrixU();

	// create eigen representation of labels and mean vectors
	SGVector<float64_t> y=((CRegressionLabels*) m_labels)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);
	SGVector<float64_t> m=m_mean->get_mean_vector(m_feat);
	Map<VectorXd> eigen_m(m.vector, m.vlen);

	// compute sgrt_dg = sqrt(dg)
	VectorXd sqrt_dg(m_ktrtr.num_cols);

	for (index_t i=0; i<m_dg.vlen; i++)
		sqrt_dg[i]=sqrt(m_dg[i]);

	// create shogun and eigen3 representation of labels adjusted for
	// noise and means (m_r)
	m_r=SGVector<float64_t>(y.vlen);
	Map<VectorXd> eigen_r(m_r.vector, m_r.vlen);
	eigen_r=(eigen_y-eigen_m).cwiseQuotient(sqrt_dg);

	// compute be
	m_be=SGVector<float64_t>(m_r.vlen);
	Map<VectorXd> eigen_be(m_be.vector, m_be.vlen);
	eigen_be=eigen_chol_utr.triangularView<Upper>().adjoint().solve(
		V*eigen_r.cwiseQuotient(sqrt_dg));

	// compute iKuu
	MatrixXd iKuu = Luu.solve(MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols));

	// create shogun and eigen3 representation of posterior cholesky
	MatrixXd eigen_prod=eigen_chol_utr*eigen_chol_uu;
	m_L = SGMatrix<float64_t>(m_kuu.num_rows, m_kuu.num_cols);
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
	m_alpha = SGVector<float64_t>(m_chol_uu.num_rows);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	eigen_alpha=eigen_chol_utr.triangularView<Upper>().solve(eigen_be);
	eigen_alpha=eigen_chol_uu.triangularView<Upper>().solve(eigen_alpha);
}

#endif // HAVE_EIGEN3
