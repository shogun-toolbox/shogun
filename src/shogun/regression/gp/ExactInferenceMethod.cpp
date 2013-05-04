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
#ifdef HAVE_EIGEN3

#include <shogun/regression/gp/ExactInferenceMethod.h>
#include <shogun/regression/gp/GaussianLikelihood.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/CombinedFeatures.h>
using namespace shogun;

#include <Eigen/Dense>
using namespace Eigen;


CExactInferenceMethod::CExactInferenceMethod() : CInferenceMethod()
{
	update_all();
	update_parameter_hash();
}

CExactInferenceMethod::CExactInferenceMethod(CKernel* kern, CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod) :
		CInferenceMethod(kern, feat, m, lab, mod)
{
	update_all();
}

CExactInferenceMethod::~CExactInferenceMethod()
{
}

void CExactInferenceMethod::update_all()
{
	if (m_labels)
		m_label_vector =
				((CRegressionLabels*) m_labels)->get_labels().clone();

	if (m_features && m_features->has_property(FP_DOT) && m_features->get_num_vectors())
		m_feature_matrix =
				((CDotFeatures*)m_features)->get_computed_dot_feature_matrix();

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
		update_chol();
		update_alpha();
	}
}

void CExactInferenceMethod::check_members()
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

	if (m_model->get_model_type() != LT_GAUSSIAN)
	{
		SG_ERROR("Exact Inference Method can only use " \
				"Gaussian Likelihood Function.\n");
	}
}

CMap<TParameter*, SGVector<float64_t> > CExactInferenceMethod::
get_marginal_likelihood_derivatives(CMap<TParameter*,
		CSGObject*>& para_dict)
{
	check_members();

	if(update_parameter_hash())
		update_all();

	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	// create eigen representation of derivative matrix and cholesky
	MatrixXd eigen_Q(m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	// solve L * L' * Q = I
	eigen_Q=eigen_L.triangularView<Upper>().adjoint().solve(
		MatrixXd::Identity(m_L.num_rows, m_L.num_cols));
	eigen_Q=eigen_L.triangularView<Upper>().solve(eigen_Q);

	// divide Q by sigma^2
	eigen_Q/=(sigma*sigma);

	// create eigen representation of alpha and compute Q = Q - alpha * alpha'
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	eigen_Q-=eigen_alpha*eigen_alpha.transpose();

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
			SGVector<float64_t> mean_derivatives;

			if (param->m_datatype.m_ctype == CT_VECTOR ||
					param->m_datatype.m_ctype == CT_SGVECTOR)
			{
				deriv = m_kernel->get_parameter_gradient(param, obj, g);
				mean_derivatives = m_mean->get_parameter_derivative(
						param, obj, m_feature_matrix, g);
			}
			else
			{
				mean_derivatives = m_mean->get_parameter_derivative(
						param, obj, m_feature_matrix);

				deriv = m_kernel->get_parameter_gradient(param, obj);
			}

			if (deriv.num_cols*deriv.num_rows > 0)
			{
				Map<MatrixXd> eigen_deriv(deriv.matrix, deriv.num_rows, deriv.num_cols);
				MatrixXd eigen_S=eigen_Q.cwiseProduct(eigen_deriv)*m_scale*m_scale;
				variables[g]=eigen_S.sum()/2.0;
				deriv_found = true;
			}
			else if (mean_derivatives.vlen > 0)
			{
				variables[g]=mean_derivatives.dot(mean_derivatives.vector,
						m_alpha.vector, m_alpha.vlen);
				deriv_found = true;
			}
		}

		if (deriv_found)
			gradient.add(param, variables);
	}

	TParameter* param;
	index_t index = get_modsel_param_index("scale");
	param = m_model_selection_parameters->get_parameter(index);

	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	MatrixXd eigen_S=eigen_Q.cwiseProduct(eigen_K)*m_scale*2.0;

	SGVector<float64_t> vscale(1);
	vscale[0]=eigen_S.sum()/2.0;

	gradient.add(param, vscale);
	para_dict.add(param, this);

	index = m_model->get_modsel_param_index("sigma");
	param = m_model->m_model_selection_parameters->get_parameter(index);

	SGVector<float64_t> vsigma(1);
	vsigma[0]=(sigma*sigma)*eigen_Q.trace();

	gradient.add(param, vsigma);
	para_dict.add(param, m_model);

	return gradient;
}

SGVector<float64_t> CExactInferenceMethod::get_diagonal_vector()
{
	if(update_parameter_hash())
		update_all();

	check_members();

	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	SGVector<float64_t> result =
			SGVector<float64_t>(m_features->get_num_vectors());

	result.fill_vector(result.vector, m_features->get_num_vectors(),
			1.0/sigma);

	return result;
}

float64_t CExactInferenceMethod::get_negative_marginal_likelihood()
{
	if(update_parameter_hash())
		update_all();

	float64_t result;

	result = m_label_vector.dot(m_label_vector.vector, m_alpha.vector,
			m_label_vector.vlen)/2.0;

	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	for (int i = 0; i < m_L.num_rows; i++)
		result += CMath::log(m_L(i,i));

	result+=m_L.num_rows*CMath::log(2*CMath::PI*sigma*sigma)/2.0;

	return result;
}

SGVector<float64_t> CExactInferenceMethod::get_alpha()
{
	if(update_parameter_hash())
		update_all();

	return SGVector<float64_t>(m_alpha);
}

SGMatrix<float64_t> CExactInferenceMethod::get_cholesky()
{
	if(update_parameter_hash())
		update_all();

	return SGMatrix<float64_t>(m_L);
}

void CExactInferenceMethod::update_train_kernel()
{
	m_kernel->cleanup();

	m_kernel->init(m_features, m_features);

	//K(X, X)
	SGMatrix<float64_t> kernel_matrix = m_kernel->get_kernel_matrix();

	m_ktrtr=kernel_matrix.clone();
}

void CExactInferenceMethod::update_chol()
{
	check_members();

	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	/* noise */
	float64_t noise=(m_scale*m_scale)/(sigma*sigma);

	/* check whether to allocate cholesky memory */
	if (!m_L.matrix || m_L.num_rows!=m_ktrtr.num_rows)
		m_L=SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);

	/* creates views on kernel and cholesky matrix and perform cholesky */
	Map<MatrixXd> K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> L(m_L.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	LLT<MatrixXd> llt;
	llt.compute(K*noise+MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));
	L=llt.matrixU();
}

void CExactInferenceMethod::update_alpha()
{
	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	for (int i = 0; i < m_label_vector.vlen; i++)
		m_label_vector[i] = m_label_vector[i] - m_data_means[i];

	m_alpha = SGVector<float64_t>(m_label_vector.vlen);

	/* creates views on cholesky matrix, labels, and alpha and solve system
	 * (L * L^T) * a = y for a */
	Map<VectorXd> a(m_alpha.vector, m_alpha.vlen);
	Map<VectorXd> y(m_label_vector.vector, m_label_vector.vlen);
	Map<MatrixXd> L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	a=L.triangularView<Upper>().adjoint().solve(y);
	a=L.triangularView<Upper>().solve(a);

	a/=(sigma*sigma);
}

#endif // HAVE_EIGEN3
