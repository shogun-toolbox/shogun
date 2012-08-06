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

#include <shogun/regression/gp/LaplacianInferenceMethod.h>
#include <shogun/regression/gp/GaussianLikelihood.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/CombinedFeatures.h>
#include <iostream>

using namespace shogun;
using namespace Eigen;

CLaplacianInferenceMethod::CLaplacianInferenceMethod() : CInferenceMethod()
{
	init();
	update_all();
	update_parameter_hash();
}

CLaplacianInferenceMethod::CLaplacianInferenceMethod(CKernel* kern, CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod, CFeatures* lat) :
			CInferenceMethod(kern, feat, m, lab, mod)
{
	init();
	set_latent_features(lat);
	update_all();
}

void CLaplacianInferenceMethod::init()
{
	m_latent_features = NULL;
	m_ind_noise = 1e-10;
	SG_ADD((CSGObject**)&m_latent_features, "latent_features",
			"Latent Features", MS_NOT_AVAILABLE);
}

CLaplacianInferenceMethod::~CLaplacianInferenceMethod()
{
}

void CLaplacianInferenceMethod::update_all()
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

void CLaplacianInferenceMethod::check_members()
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
		SG_ERROR("Laplacian Inference Method can only use " \
				"Gaussian Likelihood Function.\n");
	}
}

CMap<TParameter*, SGVector<float64_t> > CLaplacianInferenceMethod::
	get_marginal_likelihood_derivatives(CMap<TParameter*,
			CSGObject*>& para_dict)
{
	check_members();

	if(update_parameter_hash())
		update_all();

    MatrixXd Z(m_L.num_rows, m_L.num_cols);

    for (index_t i = 0; i < m_L.num_rows; i++)
    {
    	for (index_t j = 0; j < m_L.num_cols; j++)
    		Z(i,j) = m_L(i,j);
    }

	MatrixXd sW_temp(sW.rows(), m_ktrtr.num_cols);
	VectorXd sum(1);
	sum[0] = 0;


	for (index_t i = 0; i < sW.rows(); i++)
	{
		for (index_t j = 0; j < m_ktrtr.num_cols; j++)
			sW_temp(i,j) = sW(i,i);
	}

	VectorXd g;

	if (W.maxCoeff() < 0)
	{
		Z = -Z;

		MatrixXd A = MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols);

		A = A + (W.diagonal())*temp_kernel;

		FullPivLU<MatrixXd> lu(A);

		MatrixXd temp_matrix = lu.inverse().cwiseProduct(temp_kernel);
		VectorXd temp_sum(temp_matrix.rows());

		for (index_t i = 0; i < temp_matrix.rows(); i++)
		{
			for (index_t j = 0; j < temp_matrix.cols(); j++)
				temp_sum[i] += temp_matrix(i,j);
		}

	    g = temp_sum/2.0;
	}

	else
	{
		MatrixXd C = Z.colPivHouseholderQr().solve(sW_temp.cwiseProduct(temp_kernel));
	    Z = Z.llt().solve(sW.diagonal());
	    Z = sW.cwiseProduct(Z);

		VectorXd temp_sum(C.rows());

		for (index_t i = 0; i < C.rows(); i++)
		{
			for (index_t j = 0; j < C.cols(); j++)
				temp_sum[i] += C(i,j)*C(i,j);
		}

	    g = (temp_kernel.diagonal()-temp_sum)/2.0;
	}

	VectorXd dfhat = g.cwiseProduct(d3lp);

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

		for (index_t h = 0; h< length; h++)
		{

			SGMatrix<float64_t> deriv;
			SGVector<float64_t> mean_derivatives;
			VectorXd mean_dev_temp;
			VectorXd lik_first_deriv;
			VectorXd lik_second_deriv;

			if (param->m_datatype.m_ctype == CT_VECTOR ||
					param->m_datatype.m_ctype == CT_SGVECTOR)
			{
				deriv = m_kernel->get_parameter_gradient(param, obj);

				lik_first_deriv = m_model->get_first_derivative((CRegressionLabels*)m_labels, param, obj, function);
				lik_second_deriv = m_model->get_second_derivative((CRegressionLabels*)m_labels, param, obj, function);

				mean_derivatives = m_mean->get_parameter_derivative(
				 				param, obj, m_feature_matrix, h);

				 for (index_t d = 0; d < mean_derivatives.vlen; d++)
					 mean_dev_temp[d] = mean_derivatives[d];
			}

			else
			{
				mean_derivatives = m_mean->get_parameter_derivative(
				 				param, obj, m_feature_matrix);

				for (index_t d = 0; d < mean_derivatives.vlen; d++)
					 mean_dev_temp[d] = mean_derivatives[d];

				deriv = m_kernel->get_parameter_gradient(param, obj);

				lik_first_deriv = m_model->get_first_derivative((CRegressionLabels*)m_labels, param, obj, function);
				lik_second_deriv = m_model->get_second_derivative((CRegressionLabels*)m_labels, param, obj, function);
			}

			if (deriv.num_cols*deriv.num_rows > 0)
			{
				MatrixXd dK(deriv.num_cols, deriv.num_rows);

				for (index_t d = 0; d < deriv.num_rows; d++)
				{
					for (index_t s = 0; s < deriv.num_cols; s++)
						dK(d,s) = deriv(d,s);
				}

			    sum[0] = (Z.transpose()*dK).sum();
			    sum = sum - temp_alpha.transpose()*dK*temp_alpha/2.0;
			    VectorXd b = dK*dlp;
			    sum = sum - dfhat.transpose()*(b-temp_kernel*(Z*b));
				variables[h] = sum[0];
				deriv_found = true;
			}

			else if (mean_derivatives.vlen > 0)
			{
			    sum = -temp_alpha.transpose()*mean_dev_temp;
			    sum = sum - dfhat.transpose()*(mean_dev_temp-temp_kernel*
			    		(Z*mean_dev_temp));
				variables[h] = sum[0];
				deriv_found = true;
			}

			else if (lik_first_deriv.rows()*lik_second_deriv.rows() > 0)
			{
			    sum = -g.transpose()*lik_second_deriv;
			    sum[0] = sum[0] - lik_first_deriv.sum();
				variables[h] = sum[0];
				deriv_found = true;
			}

		}

		if (deriv_found)
			gradient.add(param, variables);

	}

	return gradient;
}

SGVector<float64_t> CLaplacianInferenceMethod::get_diagonal_vector()
{
	SGVector<float64_t> result(sW.rows());

	for (index_t i = 0; i < sW.rows(); i++)
		result[i] = sW(i,i);

	return result;
}

float64_t CLaplacianInferenceMethod::get_negative_marginal_likelihood()
{
	if(update_parameter_hash())
		update_all();

	if (W.maxCoeff() < 0)
	{
		MatrixXd A = MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols);

		A = A + W.diagonal()*temp_kernel;

		FullPivLU<MatrixXd> lu(A);

		float64_t result = (temp_alpha.transpose()*(function-m_means))[0]/2.0 -
			lp + log(lu.determinant())/2.0;

		return result;
	}

	else
	{
		LLT<MatrixXd> L((sW*sW.transpose()).cwiseProduct(temp_kernel) +
					MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));

		MatrixXd chol = L.matrixL();

		float64_t sum = 0;

		for (index_t i = 0; i < m_L.num_rows; i++)
			sum += log(m_L(i,i));

		float64_t result = (temp_alpha.transpose()*(function-m_means))[0]/2.0 -
			lp + sum;

		return result;
	}

}

SGVector<float64_t> CLaplacianInferenceMethod::get_alpha()
{
	if(update_parameter_hash())
		update_all();

	SGVector<float64_t> result(m_alpha);
	return result;
}

SGMatrix<float64_t> CLaplacianInferenceMethod::get_cholesky()
{
	if(update_parameter_hash())
		update_all();

	SGMatrix<float64_t> result(m_L);
	return result;
}

void CLaplacianInferenceMethod::update_train_kernel()
{
	m_kernel->cleanup();

	m_kernel->init(m_features, m_features);

	//K(X, X)
	SGMatrix<float64_t> kernel_matrix = m_kernel->get_kernel_matrix();

	m_ktrtr=kernel_matrix.clone();

	temp_kernel = MatrixXd(kernel_matrix.num_rows, kernel_matrix.num_cols);

	for (index_t i = 0; i < kernel_matrix.num_rows; i++)
	{
		for (index_t j = 0; j < kernel_matrix.num_cols; j++)
			temp_kernel(i,j) = kernel_matrix(i,j);
	}
}


void CLaplacianInferenceMethod::update_chol()
{
	check_members();

	if (W.maxCoeff() < 0)
	{
		MatrixXd A = MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols)
						+W.diagonal()*temp_kernel;

		FullPivLU<MatrixXd> lu(A);

		MatrixXd chol = -W.diagonal()*lu.inverse();

		m_L = SGMatrix<float64_t>(chol.rows(), chol.cols());

		for (index_t i = 0; i < chol.rows(); i++)
		{
			for (index_t j = 0; j < chol.cols(); j++)
				m_L(i,j) = chol(i,j);
		}

	}

	else
	{
		LLT<MatrixXd> L((sW*sW.transpose()).cwiseProduct(temp_kernel) +
					MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));

		MatrixXd chol = L.matrixL();

		m_L = SGMatrix<float64_t>(chol.rows(), chol.cols());

		for (index_t i = 0; i < chol.rows(); i++)
		{
			for (index_t j = 0; j < chol.cols(); j++)
				m_L(i,j) = chol(i,j);
		}
	}
}

void CLaplacianInferenceMethod::update_alpha()
{
	float64_t Psi_Old = CMath::INFTY;
	float64_t Psi_New;
	float64_t Psi_Def;

	SGVector<float64_t> temp_mean = m_mean->get_mean_vector(m_feature_matrix);

	m_means = VectorXd(temp_mean.vlen);
	temp_kernel = MatrixXd(m_ktrtr.num_rows, m_ktrtr.num_cols);
	temp_alpha = VectorXd(m_alpha.vlen);
	VectorXd first_derivative;

	for (index_t i = 0; i < temp_mean.vlen; i++)
		m_means[i] = temp_mean[i];

	for (index_t i = 0; i < m_alpha.vlen; i++)
		temp_alpha[i] = m_alpha[i];

	for (index_t i = 0; i < m_ktrtr.num_rows; i++)
	{
		for (index_t j = 0; j < m_ktrtr.num_cols; j++)
			temp_kernel(i,j) = m_ktrtr(i,j);
	}

	if (m_alpha.vlen != m_labels->get_num_labels())
	{
		temp_alpha = temp_alpha.Zero(m_labels->get_num_labels());

		function = temp_kernel*temp_alpha+m_means;

		W = -m_model->get_log_probability_derivative_f((CRegressionLabels*)m_labels, function, 2);
		Psi_New = -m_model->get_log_probability_f((CRegressionLabels*)m_labels, function);
	}

	else
	{
		function = temp_kernel*temp_alpha+m_means;

		W = -m_model->get_log_probability_derivative_f((CRegressionLabels*)m_labels, function, 2);
		Psi_New = (temp_alpha.transpose()*(function-m_means))[0]/2.0;
		Psi_New -= -m_model->get_log_probability_f((CRegressionLabels*)m_labels, function);
		Psi_Def = -m_model->get_log_probability_f((CRegressionLabels*)m_labels, m_means);

		if (Psi_Def < Psi_New)
		{
			temp_alpha = temp_alpha.Zero(m_labels->get_num_labels());

			W = -m_model->get_log_probability_derivative_f((CRegressionLabels*)m_labels, function, 2);
			Psi_New = -m_model->get_log_probability_f((CRegressionLabels*)m_labels, function);
		}
	}

	index_t itr = 0;

	first_derivative = m_model->get_log_probability_derivative_f((CRegressionLabels*)m_labels, function, 1);

	while (Psi_Old - Psi_New > m_tolerance && itr < m_max_itr)
	{
		Psi_Old = Psi_New;
		itr++;
		if (W.minCoeff() < 0)
		{
			float64_t coeff = W.maxCoeff();

			if (coeff < 0) coeff = 0;
			for (index_t  i = 0; i < W.rows(); i++)
				W(i) = coeff;

			m_tolerance = 1e-10;
		}

		sW = W;

		for (index_t i = 0; i < sW.rows(); i++)
		{
			for (index_t j = 0; j < sW.cols(); j++)
				sW(i,j) = CMath::sqrt(float64_t(W(i,j)));
		}


		LLT<MatrixXd> L((sW*sW.transpose()).cwiseProduct(temp_kernel) +
					MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));

		MatrixXd chol = L.matrixL();

		VectorXd b = W.cwiseProduct((function - m_means)) + first_derivative;

		chol = chol.colPivHouseholderQr().solve(sW.cwiseProduct((temp_kernel*b)));

		VectorXd dalpha = b - sW.cwiseProduct(chol) - temp_alpha;

		Psi_line func;

		func.K = &temp_kernel;
		func.alpha = &temp_alpha;
		func.dalpha = &dalpha;
		func.l1 = &lp;
		func.dl1 = &dlp;
		func.dl2 = &d2lp;
		func.f = &function;
		func.lik = m_model;
		func.m = &m_means;
		func.mW = &W;

		brent::local_min(0, m_max, m_opt_tolerance, func, Psi_New);

	}

	W = -m_model->get_log_probability_derivative_f((CRegressionLabels*)m_labels, function, 2);

	m_alpha = SGVector<float64_t>(temp_alpha.rows());
	for (index_t i = 0; i < m_alpha.vlen; i++)
		m_alpha[i] = temp_alpha[i];

}

#endif // HAVE_EIGEN3
#endif // HAVE_LAPACK

