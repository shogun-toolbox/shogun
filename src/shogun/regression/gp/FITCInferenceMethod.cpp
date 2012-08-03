/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 *
 * Code adapted from Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 *
 */
#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK
#ifdef HAVE_EIGEN3

#include <shogun/regression/gp/FITCInferenceMethod.h>
#include <shogun/regression/gp/GaussianLikelihood.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/CombinedFeatures.h>
#include <iostream>

using namespace shogun;
using namespace Eigen;

CFITCInferenceMethod::CFITCInferenceMethod() : CInferenceMethod()
{
	init();
	update_all();
	update_parameter_hash();
}

CFITCInferenceMethod::CFITCInferenceMethod(CKernel* kern, CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod, CFeatures* lat) :
			CInferenceMethod(kern, feat, m, lab, mod)
{
	init();
	set_latent_features(lat);
	update_all();
}

void CFITCInferenceMethod::init()
{
	m_latent_features = NULL;
	m_ind_noise = 1e-10;
	SG_ADD((CSGObject**)&m_latent_features, "latent_features",
			"Latent Features", MS_NOT_AVAILABLE);
}

CFITCInferenceMethod::~CFITCInferenceMethod()
{
}

void CFITCInferenceMethod::update_all()
{
	if (m_labels)
		m_label_vector =
				((CRegressionLabels*) m_labels)->get_labels().clone();

	if (m_features && m_features->has_property(FP_DOT)
			&& m_features->get_num_vectors())
	{
		m_feature_matrix =
				((CDotFeatures*)m_features)->get_computed_dot_feature_matrix();
	}

	else if (m_features && m_features->get_feature_class() == C_COMBINED)
	{
		CDotFeatures* feat =
				(CDotFeatures*)((CCombinedFeatures*)m_features)->
				get_first_feature_obj();

		if (feat->get_num_vectors())
			m_feature_matrix = feat->get_computed_dot_feature_matrix();

		SG_UNREF(feat);
	}

	if (m_latent_features && m_latent_features->has_property(FP_DOT) &&
			m_latent_features->get_num_vectors())
	{
		m_latent_matrix =
				((CDotFeatures*)m_latent_features)->
				get_computed_dot_feature_matrix();
	}

	else if (m_latent_features &&
			m_latent_features->get_feature_class() == C_COMBINED)
	{
		CDotFeatures* subfeat =
				(CDotFeatures*)((CCombinedFeatures*)m_latent_features)->
				get_first_feature_obj();

		if (m_latent_features->get_num_vectors())
			m_latent_matrix = subfeat->get_computed_dot_feature_matrix();

		SG_UNREF(subfeat);
	}

	update_data_means();

	if (m_kernel)
		update_train_kernel();

	if (m_ktrtr.num_cols*m_ktrtr.num_rows &&
		m_kuu.rows()*m_kuu.cols() &&
		m_ktru.cols()*m_ktru.rows())
	{
		update_chol();
		update_alpha();
	}
}

void CFITCInferenceMethod::check_members()
{
	if (!m_labels)
		SG_ERROR("No labels set\n");

	if (m_labels->get_label_type() != LT_REGRESSION)
		SG_ERROR("Expected RegressionLabels\n");

	if (!m_features)
		SG_ERROR("No features set!\n");

	if (!m_latent_features)
		SG_ERROR("No latent features set!\n");

  	if (m_labels->get_num_labels() != m_features->get_num_vectors())
		SG_ERROR("Number of training vectors does not match number of labels\n");

	if(m_features->get_feature_class() == C_COMBINED)
	{
		CDotFeatures* feat =
				(CDotFeatures*)((CCombinedFeatures*)m_features)->
				get_first_feature_obj();

		if (!feat->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CFeatures\n");

		if (feat->get_feature_class() != C_DENSE)
			SG_ERROR("Expected Simple Features\n");

		if (feat->get_feature_type() != F_DREAL)
			SG_ERROR("Expected Real Features\n");

		SG_UNREF(feat);
	}

	else
	{
		if (!m_features->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CFeatures\n");

		if (m_features->get_feature_class() != C_DENSE)
			SG_ERROR("Expected Simple Features\n");

		if (m_features->get_feature_type() != F_DREAL)
			SG_ERROR("Expected Real Features\n");
	}

	if(m_latent_features->get_feature_class() == C_COMBINED)
	{
		CDotFeatures* feat =
				(CDotFeatures*)((CCombinedFeatures*)m_latent_features)->
				get_first_feature_obj();

		if (!feat->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CFeatures\n");

		if (feat->get_feature_class() != C_DENSE)
			SG_ERROR("Expected Simple Features\n");

		if (feat->get_feature_type() != F_DREAL)
			SG_ERROR("Expected Real Features\n");

		SG_UNREF(feat);
	}

	else
	{
		if (!m_latent_features->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CFeatures\n");

		if (m_latent_features->get_feature_class() != C_DENSE)
			SG_ERROR("Expected Simple Features\n");

		if (m_latent_features->get_feature_type() != F_DREAL)
			SG_ERROR("Expected Real Features\n");
	}

	if (m_latent_matrix.num_rows != m_feature_matrix.num_rows)
		SG_ERROR("Regular and Latent Features do not match in dimensionality!\n");

	if (!m_kernel)
		SG_ERROR( "No kernel assigned!\n");

	if (!m_mean)
		SG_ERROR( "No mean function assigned!\n");

	if (m_model->get_model_type() != LT_GAUSSIAN)
	{
		SG_ERROR("FITC Inference Method can only use " \
				"Gaussian Likelihood Function.\n");
	}
}

CMap<TParameter*, SGVector<float64_t> > CFITCInferenceMethod::
	get_marginal_likelihood_derivatives(CMap<TParameter*,
			CSGObject*>& para_dict)
{
	check_members();

	if(update_parameter_hash())
		update_all();

	//Get the sigma variable from the likelihood model
	float64_t m_sigma =
			dynamic_cast<CGaussianLikelihood*>(m_model)->get_sigma();

	MatrixXd W = m_ktru;

	for (index_t j = 0; j < m_ktru.rows(); j++)
	{
		for (index_t i = 0; i < m_ktru.cols(); i++)
			W(i,j) = m_ktru(i,j) / sqrt(m_dg[j]);
	}

	LLT<MatrixXd> CholW(m_kuu + W*W.transpose() +
			m_ind_noise*MatrixXd::Identity(m_kuu.rows(), m_kuu.cols()));
	W = CholW.matrixL();


	W = W.colPivHouseholderQr().solve(m_ktru);

	VectorXd true_lab(m_data_means.vlen);

	for (index_t j = 0; j < m_data_means.vlen; j++)
		true_lab[j] = m_label_vector[j] - m_data_means[j];

	VectorXd al = W*true_lab.cwiseQuotient(m_dg);

	al = W.transpose()*al;

	al = true_lab - al;

	al = al.cwiseQuotient(m_dg);

	MatrixXd iKuu = m_kuu.selfadjointView<Eigen::Upper>().llt()
			.solve(MatrixXd::Identity(m_kuu.rows(), m_kuu.cols()));

	MatrixXd B = iKuu*m_ktru;

	MatrixXd Wdg = W;

	for (index_t j = 0; j < m_ktru.rows(); j++)
	{
		for (index_t i = 0; i < m_ktru.cols(); i++)
			Wdg(i,j) = Wdg(i,j) / m_dg[j];
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

				 mean_derivatives = m_mean->get_parameter_derivative(
				 				param, obj, m_feature_matrix, g);

				 for (index_t d = 0; d < mean_derivatives.vlen; d++)
					 mean_dev_temp[d] = mean_derivatives[d];
			}

			else
			{
				mean_derivatives = m_mean->get_parameter_derivative(
				 				param, obj, m_feature_matrix);

				for (index_t d = 0; d < mean_derivatives.vlen; d++)
					 mean_dev_temp[d] = mean_derivatives[d];

				m_kernel->init(m_features, m_features);
				deriv = m_kernel->get_parameter_gradient(param, obj);

				m_kernel->init(m_latent_features, m_features);
				derivtru = m_kernel->get_parameter_gradient(param, obj);

				m_kernel->init(m_latent_features, m_latent_features);
				derivuu = m_kernel->get_parameter_gradient(param, obj);
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
						ddiagKi(d,s) = deriv(d,s);
				}

				for (index_t d = 0; d < derivuu.num_rows; d++)
				{
					for (index_t s = 0; s < derivuu.num_cols; s++)
						dKuui(d,s) = derivuu(d,s);
				}

				for (index_t d = 0; d < derivtru.num_rows; d++)
				{
					for (index_t s = 0; s < derivtru.num_cols; s++)
						dKui(d,s) = derivtru(d,s);
				}

			    MatrixXd R = 2*dKui-dKuui*B;
			    MatrixXd v = ddiagKi;
			    MatrixXd temp = R.cwiseProduct(B);

			    for (index_t d = 0; d < ddiagKi.rows(); d++)
			    	v(d,d) = v(d,d) - temp.col(d).sum();

			    sum = sum + ddiagKi.diagonal().transpose()*
			    		VectorXd::Ones(m_dg.rows()).cwiseQuotient(m_dg);

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

	TParameter* param;
	index_t index;

	index = m_model->get_modsel_param_index("sigma");
	param = m_model->m_model_selection_parameters->get_parameter(index);

	sum[0] = 0;

	MatrixXd W_temp = W.cwiseProduct(W);
	VectorXd W_sum(W_temp.rows());

    for (index_t d = 0; d < W_sum.rows(); d++)
    	W_sum[d] = W_temp.col(d).sum();

    W_sum = W_sum.cwiseQuotient(m_dg.cwiseProduct(m_dg));

    sum[0] = W_sum.sum();

    sum = sum + al.transpose()*al;

	sum[0] = VectorXd::Ones(m_dg.rows()).cwiseQuotient(m_dg).sum() - sum[0];

	sum = sum*m_sigma*m_sigma;
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

	SGVector<float64_t> sigma(1);

	sigma[0] = sum[0];
	gradient.add(param, sigma);
	para_dict.add(param, m_model);

	return gradient;

}

SGVector<float64_t> CFITCInferenceMethod::get_diagonal_vector()
{
	SGVector<float64_t> result;

	return result;
}

float64_t CFITCInferenceMethod::get_negative_marginal_likelihood()
{
	if(update_parameter_hash())
		update_all();

	VectorXd temp = m_dg;
	VectorXd temp2(m_chol_utr.cols());

	for (index_t i = 0; i < m_dg.rows(); i++)
		temp[i] = log(m_dg[i]);

	for (index_t j = 0; j < m_chol_utr.rows(); j++)
		temp2[j] = log(m_chol_utr(j,j));

	VectorXd sum(1);

	sum[0] = temp.sum();
	sum = sum + m_r.transpose()*m_r;
	sum = sum - m_be.transpose()*m_be;
	sum[0] += m_label_vector.vlen*log(2*CMath::PI);
	sum /= 2.0;
	sum[0] += temp2.sum();

	return sum[0];
}

SGVector<float64_t> CFITCInferenceMethod::get_alpha()
{
	if(update_parameter_hash())
		update_all();

	SGVector<float64_t> result(m_alpha);
	return result;
}

SGMatrix<float64_t> CFITCInferenceMethod::get_cholesky()
{
	if(update_parameter_hash())
		update_all();

	SGMatrix<float64_t> result(m_L);
	return result;
}

void CFITCInferenceMethod::update_train_kernel()
{
	m_kernel->cleanup();

	m_kernel->init(m_features, m_features);

	//K(X, X)
	SGMatrix<float64_t> kernel_matrix = m_kernel->get_kernel_matrix();

	m_ktrtr=kernel_matrix.clone();

	m_kernel->cleanup();

	m_kernel->init(m_latent_features, m_latent_features);

	//K(X, X)
	kernel_matrix = m_kernel->get_kernel_matrix();

	m_kuu = MatrixXd(kernel_matrix.num_rows, kernel_matrix.num_cols);

	for (index_t i = 0; i < kernel_matrix.num_rows; i++)
	{
		for (index_t j = 0; j < kernel_matrix.num_cols; j++)
			m_kuu(i,j) = kernel_matrix(i,j);
	}

	m_kernel->cleanup();

	m_kernel->init(m_latent_features, m_features);

	kernel_matrix = m_kernel->get_kernel_matrix();

	m_ktru = MatrixXd(kernel_matrix.num_rows, kernel_matrix.num_cols);

	for (index_t i = 0; i < kernel_matrix.num_rows; i++)
	{
		for (index_t j = 0; j < kernel_matrix.num_cols; j++)
			m_ktru(i,j) = kernel_matrix(i,j);
	}
}


void CFITCInferenceMethod::update_chol()
{
	check_members();

	//Get the sigma variable from the likelihood model
	float64_t m_sigma =
			dynamic_cast<CGaussianLikelihood*>(m_model)->get_sigma();

	LLT<MatrixXd> Luu(m_kuu +
				m_ind_noise*MatrixXd::Identity(m_kuu.rows(), m_kuu.cols()));

	m_chol_uu = Luu.matrixL();

	MatrixXd V = m_chol_uu.colPivHouseholderQr().solve(m_ktru);

	MatrixXd temp_V = V.cwiseProduct(V);

	m_dg.resize(m_ktrtr.num_cols);
	VectorXd sqrt_dg(m_ktrtr.num_cols);

	for (index_t i = 0; i < m_ktrtr.num_cols; i++)
	{
		m_dg[i] = m_ktrtr(i,i) + m_sigma*m_sigma - temp_V.col(i).sum();
		sqrt_dg[i] = sqrt(m_dg[i]);
	}

	for (index_t i = 0; i < V.rows(); i++)
	{
		for (index_t j = 0; j < V.cols(); j++)
			V(i,j) /= sqrt_dg[j];
	}

	LLT<MatrixXd> Lu(V*V.transpose() +
				MatrixXd::Identity(m_kuu.rows(), m_kuu.cols()));

	m_chol_utr = Lu.matrixL();

	VectorXd true_lab(m_data_means.vlen);

	for (index_t j = 0; j < m_data_means.vlen; j++)
		true_lab[j] = m_label_vector[j] - m_data_means[j];

	m_r = true_lab.cwiseQuotient(sqrt_dg);

	m_be = m_chol_utr.colPivHouseholderQr().solve(V*m_r);

	MatrixXd iKuu = m_chol_uu.llt().solve(
			MatrixXd::Identity(m_kuu.rows(), m_kuu.cols()));

	MatrixXd chol = m_chol_utr*m_chol_uu;

	chol *= chol.transpose();

	chol = chol.llt().solve(MatrixXd::Identity(m_kuu.rows(), m_kuu.cols()));

	chol = chol - iKuu;

	m_L = SGMatrix<float64_t>(chol.rows(), chol.cols());

	for (index_t i = 0; i < chol.rows(); i++)
	{
		for (index_t j = 0; j < chol.cols(); j++)
			m_L(i,j) = chol(i,j);
	}
}

void CFITCInferenceMethod::update_alpha()
{
	MatrixXd alph;

	alph = m_chol_utr.colPivHouseholderQr().solve(m_be);

	alph = m_chol_uu.colPivHouseholderQr().solve(alph);

	m_alpha = SGVector<float64_t>(alph.rows());

	for (index_t i = 0; i < alph.rows(); i++)
		m_alpha[i] = alph(i,0);
}

#endif // HAVE_EIGEN3
#endif // HAVE_LAPACK
