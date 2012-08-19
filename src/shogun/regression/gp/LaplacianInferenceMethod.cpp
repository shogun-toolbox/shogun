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
 * This code specifically adapted from infLaplace.m
 *
 */
#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK
#ifdef HAVE_EIGEN3

#include <shogun/regression/gp/LaplacianInferenceMethod.h>
#include <shogun/regression/gp/GaussianLikelihood.h>
#include <shogun/regression/gp/StudentsTLikelihood.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/CombinedFeatures.h>

using namespace shogun;
using namespace Eigen;

CLaplacianInferenceMethod::CLaplacianInferenceMethod() : CInferenceMethod()
{
	init();
	update_all();
	update_parameter_hash();
}

CLaplacianInferenceMethod::CLaplacianInferenceMethod(CKernel* kern,
		CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod) :
		CInferenceMethod(kern, feat, m, lab, mod)
{
	init();
	update_all();
}

void CLaplacianInferenceMethod::init()
{
	m_latent_features = NULL;
	m_max_itr = 30;
	m_opt_tolerance = 1e-4;
	m_tolerance = 1e-6;
	m_max = 2;
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

	update_data_means();

	if (m_kernel)
		update_train_kernel();

	if (m_ktrtr.num_cols*m_ktrtr.num_rows)
	{
		update_alpha();
		update_chol();
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

	if (!m_kernel)
		SG_ERROR( "No kernel assigned!\n");

	if (!m_mean)
		SG_ERROR( "No mean function assigned!\n");

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
			sW_temp(i,j) = sW(i);
	}

	VectorXd g;

	if (W.minCoeff() < 0)
	{
		Z = -Z;

		MatrixXd A = MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols);

		MatrixXd temp_diagonal(sW.rows(), sW.rows());
		temp_diagonal.setZero(sW.rows(), sW.rows());

		for (index_t s = 0; s < temp_diagonal.rows(); s++)
		{
			for (index_t r = 0; r < temp_diagonal.cols(); r++)
				temp_diagonal(r,s) = W(s);
		}

		A = A + temp_kernel*m_scale*m_scale*temp_diagonal;

		FullPivLU<MatrixXd> lu(A);

		MatrixXd temp_matrix =
				lu.inverse().cwiseProduct(temp_kernel*m_scale*m_scale);

		VectorXd temp_sum(temp_matrix.rows());

		for (index_t i = 0; i < temp_matrix.rows(); i++)
		{
			for (index_t j = 0; j < temp_matrix.cols(); j++)
				temp_sum[i] += temp_matrix(j,i);
		}

		g = temp_sum/2.0;
	}

	else
	{
		MatrixXd C = Z.transpose().colPivHouseholderQr().solve(
				sW_temp.cwiseProduct(temp_kernel*m_scale*m_scale));

		MatrixXd temp_diagonal(sW.rows(), sW.rows());
		temp_diagonal.setZero(sW.rows(), sW.rows());

		for (index_t s = 0; s < sW.rows(); s++)
			temp_diagonal(s,s) = sW(s);

		MatrixXd temp = Z.transpose();

		Z = Z.transpose().colPivHouseholderQr().solve(temp_diagonal);

		Z = temp.transpose().colPivHouseholderQr().solve(Z);

		for (index_t s = 0; s < Z.rows(); s++)
		{
			for (index_t r = 0; r < Z.cols(); r++)
				Z(s,r) *= sW(s);
		}

		VectorXd temp_sum(C.rows());

		temp_sum.setZero(C.rows());

		for (index_t i = 0; i < C.rows(); i++)
		{
			for (index_t j = 0; j < C.cols(); j++)
				temp_sum[i] += C(j,i)*C(j,i);
		}

		g = (temp_kernel.diagonal()*m_scale*m_scale-temp_sum)/2.0;
	}

	VectorXd dfhat = g.cwiseProduct(d3lp);

	m_kernel->build_parameter_dictionary(para_dict);
	m_mean->build_parameter_dictionary(para_dict);
	m_model->build_parameter_dictionary(para_dict);

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

		for (index_t h = 0; h < length; h++)
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

				lik_first_deriv = m_model->get_first_derivative(
						(CRegressionLabels*)m_labels, param, obj, function);

				lik_second_deriv = m_model->get_second_derivative(
						(CRegressionLabels*)m_labels, param, obj, function);

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

				lik_first_deriv = m_model->get_first_derivative(
						(CRegressionLabels*)m_labels, param, obj, function);

				lik_second_deriv = m_model->get_second_derivative(
						(CRegressionLabels*)m_labels, param, obj, function);
			}

			if (deriv.num_cols*deriv.num_rows > 0)
			{
				MatrixXd dK(deriv.num_cols, deriv.num_rows);

				for (index_t d = 0; d < deriv.num_rows; d++)
				{
					for (index_t s = 0; s < deriv.num_cols; s++)
						dK(d,s) = deriv(d,s);
				}


				sum[0] = (Z.cwiseProduct(dK)).sum()/2.0;


				sum = sum - temp_alpha.transpose()*dK*temp_alpha/2.0;

				VectorXd b = dK*dlp;

				sum = sum -
						dfhat.transpose()*(b-temp_kernel*(Z*b)*m_scale*m_scale);

				variables[h] = sum[0];

				deriv_found = true;
			}

			else if (mean_derivatives.vlen > 0)
			{
				sum = -temp_alpha.transpose()*mean_dev_temp;
				sum = sum - dfhat.transpose()*(mean_dev_temp-temp_kernel*
						(Z*mean_dev_temp)*m_scale*m_scale);
				variables[h] = sum[0];
				deriv_found = true;
			}

			else if (lik_first_deriv[0]+lik_second_deriv[0] != CMath::INFTY)
			{
				sum[0] = -g.dot(lik_second_deriv);
				sum[0] = sum[0] - lik_first_deriv.sum();
				variables[h] = sum[0];
				deriv_found = true;
			}

		}

		if (deriv_found)
			gradient.add(param, variables);

	}

	TParameter* param;
	index_t index = get_modsel_param_index("scale");
	param = m_model_selection_parameters->get_parameter(index);

	MatrixXd dK(m_ktrtr.num_cols, m_ktrtr.num_rows);

	for (index_t d = 0; d < m_ktrtr.num_rows; d++)
	{
		for (index_t s = 0; s < m_ktrtr.num_cols; s++)
			dK(d,s) = m_ktrtr(d,s)*m_scale*2.0;;
	}

	sum[0] = (Z.cwiseProduct(dK)).sum()/2.0;

	sum = sum - temp_alpha.transpose()*dK*temp_alpha/2.0;

	VectorXd b = dK*dlp;

	sum = sum - dfhat.transpose()*(b-temp_kernel*(Z*b)*m_scale*m_scale);

	SGVector<float64_t> scale(1);

	scale[0] = sum[0];

	gradient.add(param, scale);
	para_dict.add(param, this);

	return gradient;
}

SGVector<float64_t> CLaplacianInferenceMethod::get_diagonal_vector()
{
	SGVector<float64_t> result(sW.rows());

	for (index_t i = 0; i < sW.rows(); i++)
		result[i] = sW(i);

	return result;
}

float64_t CLaplacianInferenceMethod::get_negative_marginal_likelihood()
{
	if(update_parameter_hash())
		update_all();

	if (W.minCoeff() < 0)
	{
		MatrixXd A = MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols);

		MatrixXd temp_diagonal(sW.rows(), sW.rows());
		temp_diagonal.setZero(sW.rows(), sW.rows());

		for (index_t s = 0; s < sW.rows(); s++)
			temp_diagonal(s,s) = sW(s);

		A = A + temp_kernel*m_scale*m_scale*temp_diagonal;

		FullPivLU<MatrixXd> lu(A);

		float64_t result = (temp_alpha.transpose()*(function-m_means))[0]/2.0 -
				lp + log(lu.determinant())/2.0;

		return result;
	}

	else
	{
		LLT<MatrixXd> L(
				(sW*sW.transpose()).cwiseProduct(temp_kernel*m_scale*m_scale) +
				MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));

		MatrixXd chol = L.matrixL();

		float64_t sum = 0;

		for (index_t i = 0; i < m_L.num_rows; i++)
			sum += log(m_L(i,i));

		float64_t result = temp_alpha.dot(function-m_means)/2.0 -
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

	if (W.minCoeff() < 0)
	{
		MatrixXd temp_diagonal(sW.rows(), sW.rows());
		temp_diagonal.setZero(sW.rows(), sW.rows());

		for (index_t s = 0; s < temp_diagonal.rows(); s++)
		{
			for (index_t r = 0; r < temp_diagonal.cols(); r++)
				temp_diagonal(s,s) = 1.0/W(s);
		}

		MatrixXd A = temp_kernel*m_scale*m_scale+temp_diagonal;

		FullPivLU<MatrixXd> lu(A);

		MatrixXd chol = -lu.inverse();

		m_L = SGMatrix<float64_t>(chol.rows(), chol.cols());

		for (index_t i = 0; i < chol.rows(); i++)
		{
			for (index_t j = 0; j < chol.cols(); j++)
				m_L(i,j) = chol(i,j);
		}

	}

	else
	{
		LLT<MatrixXd> L(
				(sW*sW.transpose()).cwiseProduct((temp_kernel*m_scale*m_scale)) +
				MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));

		MatrixXd chol = L.matrixU();

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

		function = temp_kernel*temp_alpha*m_scale*m_scale+m_means;

		W = -m_model->get_log_probability_derivative_f(
				(CRegressionLabels*)m_labels, function, 2);

		Psi_New = -m_model->get_log_probability_f(
				(CRegressionLabels*)m_labels, function);
	}


	else
	{
		function = temp_kernel*m_scale*m_scale*temp_alpha+m_means;


		W = -m_model->get_log_probability_derivative_f(
				(CRegressionLabels*)m_labels, function, 2);
		Psi_New = (temp_alpha.transpose()*(function-m_means))[0]/2.0;

		Psi_New -= -m_model->get_log_probability_f(
				(CRegressionLabels*)m_labels, function);

		Psi_Def = -m_model->get_log_probability_f(
				(CRegressionLabels*)m_labels, m_means);

		if (Psi_Def < Psi_New)
		{
			temp_alpha = temp_alpha.Zero(m_labels->get_num_labels());

			W = -m_model->get_log_probability_derivative_f(
					(CRegressionLabels*)m_labels, function, 2);

			Psi_New = -m_model->get_log_probability_f(
					(CRegressionLabels*)m_labels, function);
		}
	}

	index_t itr = 0;

	dlp = m_model->get_log_probability_derivative_f(
			(CRegressionLabels*)m_labels, function, 1);

	d2lp = m_model->get_log_probability_derivative_f(
			(CRegressionLabels*)m_labels, function, 2);

	d3lp = m_model->get_log_probability_derivative_f(
			(CRegressionLabels*)m_labels, function, 3);

	while (Psi_Old - Psi_New > m_tolerance && itr < m_max_itr)
	{
		Psi_Old = Psi_New;
		itr++;

		if (W.minCoeff() < 0)
		{
			//Suggested by Vanhatalo et. al.,
			//Gaussian Process Regression with Student's t likelihood, NIPS 2009
			//Quoted from infLaplace.m
			float64_t df = m_model->get_degrees_freedom();
			W = W + 2.0/(df)*dlp.cwiseProduct(dlp);
		}

		sW = W;

		for (index_t i = 0; i < sW.rows(); i++)
		{
			for (index_t j = 0; j < sW.cols(); j++)
				sW(i,j) = CMath::sqrt(float64_t(W(i,j)));
		}

		LLT<MatrixXd> L((sW*sW.transpose()).cwiseProduct(
				temp_kernel*m_scale*m_scale) +
				MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));

		MatrixXd chol = L.matrixL();

		MatrixXd temp = L.matrixL();

		VectorXd b = W.cwiseProduct((function - m_means)) + dlp;

		chol = chol.colPivHouseholderQr().solve(sW.cwiseProduct(
				(temp_kernel*b*m_scale*m_scale)));

		chol = temp.transpose().colPivHouseholderQr().solve(chol);

		VectorXd dalpha = b - sW.cwiseProduct(chol) - temp_alpha;
		Psi_line func;

		func.lab = (CRegressionLabels*)m_labels;
		func.K = &temp_kernel;
		func.scale = m_scale;
		func.alpha = &temp_alpha;
		func.dalpha = &dalpha;
		func.l1 = &lp;
		func.dl1 = &dlp;
		func.dl2 = &d2lp;
		func.f = &function;
		func.lik = m_model;
		func.m = &m_means;
		func.mW = &W;
		func.start_alpha = temp_alpha;
		local_min(0, m_max, m_opt_tolerance, func, Psi_New);
	}


	function = temp_kernel*m_scale*m_scale*temp_alpha+m_means;

	lp = m_model->get_log_probability_f((CRegressionLabels*)m_labels, function);

	dlp = m_model->get_log_probability_derivative_f(
			(CRegressionLabels*)m_labels, function, 1);

	d2lp = m_model->get_log_probability_derivative_f(
			(CRegressionLabels*)m_labels, function, 2);

	d3lp = m_model->get_log_probability_derivative_f(
			(CRegressionLabels*)m_labels, function, 3);

	W = -m_model->get_log_probability_derivative_f(
			(CRegressionLabels*)m_labels, function, 2);

	sW.setZero(sW.rows(), sW.cols());

	if (W.minCoeff() > 0)
	{
		for (index_t i = 0; i < sW.rows(); i++)
		{
			for (index_t j = 0; j < sW.cols(); j++)
				sW(i,j) = CMath::sqrt(float64_t(W(i,j)));
		}
	}

	m_alpha = SGVector<float64_t>(temp_alpha.rows());

	for (index_t i = 0; i < m_alpha.vlen; i++)
		m_alpha[i] = temp_alpha[i];

}

#endif // HAVE_EIGEN3
#endif // HAVE_LAPACK

