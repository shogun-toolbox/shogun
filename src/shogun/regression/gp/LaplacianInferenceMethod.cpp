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
#include <shogun/mathematics/eigen3.h>
#include <shogun/lib/external/brent.h>

using namespace shogun;
using namespace Eigen;

namespace shogun
{
	/*Wrapper class used for the Brent minimizer
	 *
	 */
	class Psi_line : public func_base
	{
	public:
		Eigen::Map<Eigen::VectorXd>* alpha;
		Eigen::VectorXd* dalpha;
		Eigen::Map<Eigen::MatrixXd>* K;
		float64_t* l1;
		SGVector<float64_t>* dl1;
		Eigen::Map<Eigen::VectorXd>* dl2;
		SGVector<float64_t>* mW;
		SGVector<float64_t>* f;
		SGVector<float64_t>* m;
		float64_t scale;
		CLikelihoodModel* lik;
		CRegressionLabels *lab;

		Eigen::VectorXd start_alpha;

		virtual double operator() (double x)
		{
			Eigen::Map<Eigen::VectorXd> eigen_f(f->vector, f->vlen);
			Eigen::Map<Eigen::VectorXd> eigen_m(m->vector, m->vlen);

			*alpha = start_alpha + x*(*dalpha);
			(eigen_f) = (*K)*(*alpha)*scale*scale+(eigen_m);


			for (index_t i = 0; i < eigen_f.rows(); i++)
				(*f)[i] = eigen_f[i];

			(*dl1) = lik->get_log_probability_derivative_f(lab, (*f), 1);
			(*mW) = lik->get_log_probability_derivative_f(lab, (*f), 2);
			float64_t result = ((*alpha).dot(((eigen_f)-(eigen_m))))/2.0;

			for (index_t i = 0; i < (*mW).vlen; i++)
				(*mW)[i] = -(*mW)[i];



			result -= lik->get_log_probability_f(lab, *f);

			return result;
		}
	};
}

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
	m_opt_tolerance = 1e-6;
	m_tolerance = 1e-8;
	m_max = 5;
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
		SG_ERROR("No labels set\n")

	if (m_labels->get_label_type() != LT_REGRESSION)
		SG_ERROR("Expected RegressionLabels\n")

	if (!m_features)
		SG_ERROR("No features set!\n")

	if (m_labels->get_num_labels() != m_features->get_num_vectors())
		SG_ERROR("Number of training vectors does not match number of labels\n")

	if(m_features->get_feature_class() == C_COMBINED)
	{
		CDotFeatures* feat =
				(CDotFeatures*)((CCombinedFeatures*)m_features)->
				get_first_feature_obj();

		if (!feat->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CFeatures\n")

		if (feat->get_feature_class() != C_DENSE)
			SG_ERROR("Expected Simple Features\n")

		if (feat->get_feature_type() != F_DREAL)
			SG_ERROR("Expected Real Features\n")

		SG_UNREF(feat);
	}

	else
	{
		if (!m_features->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CFeatures\n")

		if (m_features->get_feature_class() != C_DENSE)
			SG_ERROR("Expected Simple Features\n")

		if (m_features->get_feature_type() != F_DREAL)
			SG_ERROR("Expected Real Features\n")
	}

	if (!m_kernel)
		SG_ERROR("No kernel assigned!\n")

	if (!m_mean)
		SG_ERROR("No mean function assigned!\n")

}

CMap<TParameter*, SGVector<float64_t> > CLaplacianInferenceMethod::
get_marginal_likelihood_derivatives(CMap<TParameter*,
		CSGObject*>& para_dict)
{
	check_members();

	if(update_parameter_hash())
		update_all();

	MatrixXd Z(m_L.num_rows, m_L.num_cols);

	Map<VectorXd> eigen_dlp(dlp.vector, dlp.vlen);

	for (index_t i = 0; i < m_L.num_rows; i++)
	{
		for (index_t j = 0; j < m_L.num_cols; j++)
			Z(i,j) = m_L(i,j);
	}

	MatrixXd sW_temp(sW.vlen, m_ktrtr.num_cols);
	VectorXd sum(1);
	sum[0] = 0;


	for (index_t i = 0; i < sW.vlen; i++)
	{
		for (index_t j = 0; j < m_ktrtr.num_cols; j++)
			sW_temp(i,j) = sW[i];
	}

	VectorXd g;

	Map<VectorXd> eigen_W(W.vector, W.vlen);

	Map<MatrixXd> eigen_temp_kernel(temp_kernel.matrix,
        	temp_kernel.num_rows, temp_kernel.num_cols);

	if (eigen_W.minCoeff() < 0)
	{
		Z = -Z;

		MatrixXd A = MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols);

		MatrixXd temp_diagonal(sW.vlen, sW.vlen);
		temp_diagonal.setZero(sW.vlen, sW.vlen);

		for (index_t s = 0; s < temp_diagonal.rows(); s++)
		{
			for (index_t r = 0; r < temp_diagonal.cols(); r++)
				temp_diagonal(r,s) = W[s];
		}

		A = A + eigen_temp_kernel*m_scale*m_scale*temp_diagonal;

		FullPivLU<MatrixXd> lu(A);

		MatrixXd temp_matrix =
				lu.inverse().cwiseProduct(eigen_temp_kernel*m_scale*m_scale);

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
				sW_temp.cwiseProduct(eigen_temp_kernel*m_scale*m_scale));

		MatrixXd temp_diagonal(sW.vlen, sW.vlen);
		temp_diagonal.setZero(sW.vlen, sW.vlen);

		for (index_t s = 0; s < sW.vlen; s++)
			temp_diagonal(s,s) = sW[s];

		MatrixXd temp = Z.transpose();

		Z = Z.transpose().colPivHouseholderQr().solve(temp_diagonal);

		Z = temp.transpose().colPivHouseholderQr().solve(Z);

		for (index_t s = 0; s < Z.rows(); s++)
		{
			for (index_t r = 0; r < Z.cols(); r++)
				Z(s,r) *= sW[s];
		}

		VectorXd temp_sum(C.rows());

		temp_sum.setZero(C.rows());

		for (index_t i = 0; i < C.rows(); i++)
		{
			for (index_t j = 0; j < C.cols(); j++)
				temp_sum[i] += C(j,i)*C(j,i);
		}

		g = (eigen_temp_kernel.diagonal()*m_scale*m_scale-temp_sum)/2.0;
	}

	Map<VectorXd> eigen_d3lp(d3lp.vector, d3lp.vlen);

	VectorXd dfhat = g.cwiseProduct(eigen_d3lp);

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

		Map<VectorXd> eigen_temp_alpha(temp_alpha.vector,
			temp_alpha.vlen);

		for (index_t h = 0; h < length; h++)
		{

			SGMatrix<float64_t> deriv;
			SGVector<float64_t> mean_derivatives;
			VectorXd mean_dev_temp;
			SGVector<float64_t> lik_first_deriv;
			SGVector<float64_t> lik_second_deriv;

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


				sum = sum - eigen_temp_alpha.transpose()*dK*eigen_temp_alpha/2.0;

				VectorXd b = dK*eigen_dlp;

				sum = sum -
						dfhat.transpose()*(b-eigen_temp_kernel*(Z*b)*m_scale*m_scale);

				variables[h] = sum[0];

				deriv_found = true;
			}

			else if (mean_derivatives.vlen > 0)
			{
				sum = -eigen_temp_alpha.transpose()*mean_dev_temp;
				sum = sum - dfhat.transpose()*(mean_dev_temp-eigen_temp_kernel*
						(Z*mean_dev_temp)*m_scale*m_scale);
				variables[h] = sum[0];
				deriv_found = true;
			}

			else if (lik_first_deriv[0]+lik_second_deriv[0] != CMath::INFTY)
			{
				Map<VectorXd> eigen_fd(lik_first_deriv.vector, lik_first_deriv.vlen);

				Map<VectorXd> eigen_sd(lik_second_deriv.vector, lik_second_deriv.vlen);

				sum[0] = -g.dot(eigen_sd);
				sum[0] = sum[0] - eigen_fd.sum();
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

	Map<VectorXd> eigen_temp_alpha(temp_alpha.vector,
		temp_alpha.vlen);

	sum[0] = (Z.cwiseProduct(dK)).sum()/2.0;

	sum = sum - eigen_temp_alpha.transpose()*dK*eigen_temp_alpha/2.0;

	VectorXd b = dK*eigen_dlp;

	sum = sum - dfhat.transpose()*(b-eigen_temp_kernel*(Z*b)*m_scale*m_scale);

	SGVector<float64_t> scale(1);

	scale[0] = sum[0];

	gradient.add(param, scale);
	para_dict.add(param, this);

	return gradient;
}

SGVector<float64_t> CLaplacianInferenceMethod::get_diagonal_vector()
{
	SGVector<float64_t> result(sW.vlen);

	for (index_t i = 0; i < sW.vlen; i++)
		result[i] = sW[i];

	return result;
}

float64_t CLaplacianInferenceMethod::get_negative_marginal_likelihood()
{
	if(update_parameter_hash())
		update_all();

	Map<VectorXd> eigen_temp_alpha(temp_alpha.vector, temp_alpha.vlen);

	Map<MatrixXd> eigen_temp_kernel(temp_kernel.matrix,
temp_kernel.num_rows, temp_kernel.num_cols);

	Map<VectorXd> eigen_function(function.vector,
		function.vlen);

	Map<VectorXd> eigen_W(W.vector, W.vlen);

	Map<VectorXd> eigen_m_means(m_means.vector, m_means.vlen);

	if (eigen_W.minCoeff() < 0)
	{
		MatrixXd A = MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols);

		MatrixXd temp_diagonal(sW.vlen, sW.vlen);
		temp_diagonal.setZero(sW.vlen, sW.vlen);

		for (index_t s = 0; s < sW.vlen; s++)
			temp_diagonal(s,s) = sW[s];

		A = A + eigen_temp_kernel*m_scale*m_scale*temp_diagonal;

		FullPivLU<MatrixXd> lu(A);

		float64_t result = (eigen_temp_alpha.transpose()*(eigen_function-eigen_m_means))[0]/2.0 -
				lp + log(lu.determinant())/2.0;

		return result;
	}

	else
	{

		Map<VectorXd> eigen_sW(sW.vector, sW.vlen);

		LLT<MatrixXd> L(
				(eigen_sW*eigen_sW.transpose()).cwiseProduct(eigen_temp_kernel*m_scale*m_scale) +
				MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));

		MatrixXd chol = L.matrixL();

		float64_t sum = 0;

		for (index_t i = 0; i < m_L.num_rows; i++)
			sum += log(m_L(i,i));

		float64_t result = eigen_temp_alpha.dot(eigen_function-eigen_m_means)/2.0 -
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

	temp_kernel =SGMatrix<float64_t>(kernel_matrix.num_rows, kernel_matrix.num_cols);

	for (index_t i = 0; i < kernel_matrix.num_rows; i++)
	{
		for (index_t j = 0; j < kernel_matrix.num_cols; j++)
			temp_kernel(i,j) = kernel_matrix(i,j);
	}
}


void CLaplacianInferenceMethod::update_chol()
{
	check_members();

	Map<VectorXd> eigen_W(W.vector, W.vlen);

	Map<MatrixXd> eigen_temp_kernel(temp_kernel.matrix,
		temp_kernel.num_rows, temp_kernel.num_cols);


	if (eigen_W.minCoeff() < 0)
	{
		MatrixXd temp_diagonal(sW.vlen, sW.vlen);
		temp_diagonal.setZero(sW.vlen, sW.vlen);

		for (index_t s = 0; s < temp_diagonal.rows(); s++)
		{
			for (index_t r = 0; r < temp_diagonal.cols(); r++)
				temp_diagonal(s,s) = 1.0/W[s];
		}


		MatrixXd A = eigen_temp_kernel*m_scale*m_scale+temp_diagonal;

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

		Map<VectorXd> eigen_sW(sW.vector, sW.vlen);

		LLT<MatrixXd> L(
				(eigen_sW*eigen_sW.transpose()).cwiseProduct((eigen_temp_kernel*m_scale*m_scale)) +
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

	m_means = SGVector<float64_t>(temp_mean.vlen);
	temp_kernel = SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);
	temp_alpha = SGVector<float64_t>(m_labels->get_num_labels());
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

	Map<MatrixXd> eigen_temp_kernel(temp_kernel.matrix, temp_kernel.num_rows, temp_kernel.num_cols);

	VectorXd eigen_function;

	Map<VectorXd> eigen_m_means(m_means.vector, m_means.vlen);

	if (m_alpha.vlen != m_labels->get_num_labels())
	{


		temp_alpha = SGVector<float64_t>(m_labels->get_num_labels());

		for (index_t i = 0; i < temp_alpha.vlen; i++)
			temp_alpha[i] = 0.0;

		Map<VectorXd> eigen_temp_alpha2(temp_alpha.vector, temp_alpha.vlen);

		eigen_function = eigen_temp_kernel*eigen_temp_alpha2*m_scale*m_scale+eigen_m_means;

		function = SGVector<float64_t>(eigen_function.rows());

		for (index_t i = 0; i < eigen_function.rows(); i++)
			function[i] = eigen_function[i];

		W = m_model->get_log_probability_derivative_f(
				(CRegressionLabels*)m_labels, function, 2);
		for (index_t i = 0; i < W.vlen; i++)
			W[i] = -W[i];

		Psi_New = -m_model->get_log_probability_f(
				(CRegressionLabels*)m_labels, function);
	}


	else
	{

		Map<VectorXd> eigen_temp_alpha2(temp_alpha.vector, temp_alpha.vlen);

		eigen_function = eigen_temp_kernel*m_scale*m_scale*eigen_temp_alpha2+eigen_m_means;

		function = SGVector<float64_t>(eigen_function.rows());

		for (index_t i = 0; i < eigen_function.rows(); i++)
			function[i] = eigen_function[i];



		W = m_model->get_log_probability_derivative_f(
				(CRegressionLabels*)m_labels, function, 2);


		for (index_t i = 0; i < W.vlen; i++)
			W[i] = -W[i];

		Psi_New = (eigen_temp_alpha2.transpose()*(eigen_function-eigen_m_means))[0]/2.0;

		Psi_New -= m_model->get_log_probability_f(
				(CRegressionLabels*)m_labels, function);



		Psi_Def = -m_model->get_log_probability_f(
				(CRegressionLabels*)m_labels, m_means);

		if (Psi_Def < Psi_New)
		{
			eigen_temp_alpha2 = eigen_temp_alpha2.Zero(m_labels->get_num_labels());

			for (index_t i = 0; i < m_alpha.vlen; i++)
				temp_alpha[i] = eigen_temp_alpha2[i];

			W = m_model->get_log_probability_derivative_f(
					(CRegressionLabels*)m_labels, function, 2);

			for (index_t i = 0; i < W.vlen; i++)
				W[i] = -W[i];

			Psi_New = -m_model->get_log_probability_f(
					(CRegressionLabels*)m_labels, function);
		}
	}

	Map<VectorXd> eigen_temp_alpha(temp_alpha.vector, temp_alpha.vlen);

	index_t itr = 0;

	dlp = m_model->get_log_probability_derivative_f(
			(CRegressionLabels*)m_labels, function, 1);

	d2lp = m_model->get_log_probability_derivative_f(
			(CRegressionLabels*)m_labels, function, 2);

	d3lp = m_model->get_log_probability_derivative_f(
			(CRegressionLabels*)m_labels, function, 3);

	Map<VectorXd> eigen_d2lp(d2lp.vector, d2lp.vlen);

	sW = W.clone();

	while (Psi_Old - Psi_New > m_tolerance && itr < m_max_itr)
	{

		Map<VectorXd> eigen_W(W.vector, W.vlen);

		Psi_Old = Psi_New;
		itr++;

		if (eigen_W.minCoeff() < 0)
		{
			//Suggested by Vanhatalo et. al.,
			//Gaussian Process Regression with Student's t likelihood, NIPS 2009
			//Quoted from infLaplace.m
			float64_t df;

			if (m_model->get_model_type()==LT_STUDENTST)
			{
				CStudentsTLikelihood* lik=CStudentsTLikelihood::obtain_from_generic(m_model);
				df=lik->get_degrees_freedom();
			}
			else
				df=1;

			for (index_t i = 0; i < eigen_W.rows(); i++)
				eigen_W[i] += 2.0/(df)*dlp[i]*dlp[i];

		}

		for (index_t i = 0; i < eigen_W.rows(); i++)
			W[i] = eigen_W[i];

		for (index_t i = 0; i < W.vlen; i++)
			sW[i] = CMath::sqrt(float64_t(W[i]));

		Map<VectorXd> eigen_sW(sW.vector, sW.vlen);

		LLT<MatrixXd> L((eigen_sW*eigen_sW.transpose()).cwiseProduct(
				eigen_temp_kernel*m_scale*m_scale) +
				MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));

		Map<VectorXd> temp_eigen_dlp(dlp.vector, dlp.vlen);

		VectorXd b = eigen_W.cwiseProduct((eigen_function - eigen_m_means)) + temp_eigen_dlp;

		MatrixXd chol = L.solve(eigen_sW.cwiseProduct((eigen_temp_kernel*b*m_scale*m_scale)));

		VectorXd dalpha = b - eigen_sW.cwiseProduct(chol) - eigen_temp_alpha;

		Psi_line func;

		func.lab = (CRegressionLabels*)m_labels;
		func.K = &eigen_temp_kernel;
		func.scale = m_scale;
		func.alpha = &eigen_temp_alpha;
		func.dalpha = &dalpha;
		func.l1 = &lp;
		func.dl1 = &dlp;
		func.dl2 = &eigen_d2lp;
		func.f = &function;
		func.lik = m_model;
		func.m = &m_means;
		func.mW = &W;
		func.start_alpha = eigen_temp_alpha;
		local_min(0, m_max, m_opt_tolerance, func, Psi_New);
	}

	for (index_t i = 0; i < m_alpha.vlen; i++)
		temp_alpha[i] = eigen_temp_alpha[i];

	Map<VectorXd> eigen_dlp(dlp.vector, dlp.vlen);

	eigen_function = eigen_temp_kernel*m_scale*m_scale*eigen_temp_alpha+eigen_m_means;

	function = SGVector<float64_t>(eigen_function.rows());

	for (index_t i = 0; i < eigen_function.rows(); i++)
		function[i] = eigen_function[i];

	lp = m_model->get_log_probability_f((CRegressionLabels*)m_labels, function);

	dlp = m_model->get_log_probability_derivative_f(
			(CRegressionLabels*)m_labels, function, 1);

	d2lp = m_model->get_log_probability_derivative_f(
			(CRegressionLabels*)m_labels, function, 2);

	d3lp = m_model->get_log_probability_derivative_f(
			(CRegressionLabels*)m_labels, function, 3);

	W = m_model->get_log_probability_derivative_f(
			(CRegressionLabels*)m_labels, function, 2);

	for (index_t i = 0; i < W.vlen; i++)
		W[i] = -W[i];

	for (index_t i = 0; i < sW.vlen; i++)
		sW[i] = 0.0;

	Map<VectorXd> eigen_W2(W.vector, W.vlen);

	if (eigen_W2.minCoeff() > 0)
	{
		for (index_t i = 0; i < sW.vlen; i++)
			sW[i] = CMath::sqrt(float64_t(W[i]));
	}

	m_alpha = SGVector<float64_t>(eigen_temp_alpha.rows());

	for (index_t i = 0; i < m_alpha.vlen; i++)
		m_alpha[i] = eigen_temp_alpha[i];

}

#endif // HAVE_EIGEN3
#endif // HAVE_LAPACK

