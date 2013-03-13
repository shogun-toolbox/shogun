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
#ifdef HAVE_LAPACK

#include <shogun/regression/gp/ExactInferenceMethod.h>
#include <shogun/regression/gp/GaussianLikelihood.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/CombinedFeatures.h>

#include <Eigen/Dense>
#include <iostream>
using namespace std;

using namespace shogun;
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

	//Get the sigma variable from the likelihood model
	float64_t m_sigma =
			dynamic_cast<CGaussianLikelihood*>(m_model)->get_sigma();

	//Placeholder Matrix
	SGMatrix<float64_t> temp1(m_ktrtr.num_rows, m_ktrtr.num_cols);

	//Placeholder Matrix
	SGMatrix<float64_t> temp2(m_alpha.vlen, m_alpha.vlen);

	//Derivative Matrix
	SGMatrix<float64_t> Q(m_L.num_rows, m_L.num_cols);

	//Vector used to fill diagonal of Matrix.
	SGVector<float64_t> diagonal(temp1.num_rows);
	SGVector<float64_t> diagonal2(temp2.num_rows);

	diagonal.fill_vector(diagonal.vector, temp1.num_rows, 1.0);
	diagonal2.fill_vector(diagonal2.vector, temp2.num_rows, 0.0);

	temp1.create_diagonal_matrix(temp1.matrix, diagonal.vector, temp1.num_rows);
	Q.create_diagonal_matrix(Q.matrix, diagonal.vector, temp2.num_rows);
	temp2.create_diagonal_matrix(temp2.matrix, diagonal2.vector, temp2.num_rows);

	memcpy(temp1.matrix, m_L.matrix,
			m_L.num_cols*m_L.num_rows*sizeof(float64_t));

	//Solve (L) Q = Identity for Q.
	clapack_dpotrs(CblasColMajor, CblasUpper,
			temp1.num_rows, Q.num_cols, temp1.matrix, temp1.num_cols,
			Q.matrix, Q.num_cols);

	//Calculate alpha*alpha'
	cblas_dger(CblasColMajor, m_alpha.vlen, m_alpha.vlen,
			1.0, m_alpha.vector, 1, m_alpha.vector, 1,
			temp2.matrix, m_alpha.vlen);

	temp1.create_diagonal_matrix(temp1.matrix, diagonal.vector, temp1.num_rows);

	//Subtracct alpha*alpha' from Q.
	cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper,
			temp1.num_rows, temp1.num_cols, 1.0/(m_sigma*m_sigma),
			Q.matrix, temp1.num_cols,
			temp1.matrix, temp1.num_cols, -1.0,
			temp2.matrix, temp2.num_cols);

	memcpy(Q.matrix, temp2.matrix,
			temp2.num_cols*temp2.num_rows*sizeof(float64_t));

	float64_t sum = 0;

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

			sum = 0;


			if (deriv.num_cols*deriv.num_rows > 0)
			{
				for (index_t k = 0; k < Q.num_rows; k++)
				{
					for (index_t j = 0; j < Q.num_cols; j++)
						sum += Q(k,j)*deriv(k,j)*m_scale*m_scale;
				}

				sum /= 2.0;
				variables[g] = sum;
				deriv_found = true;
			}

			else if (mean_derivatives.vlen > 0)
			{
				sum = mean_derivatives.dot(mean_derivatives.vector,
						m_alpha.vector, m_alpha.vlen);
				variables[g] = sum;
				deriv_found = true;
			}


		}

		if (deriv_found)
			gradient.add(param, variables);

	}

	TParameter* param;
	index_t index = get_modsel_param_index("scale");
	param = m_model_selection_parameters->get_parameter(index);

	sum = 0;

	for (index_t i = 0; i < Q.num_rows; i++)
	{
		for (index_t j = 0; j < Q.num_cols; j++)
			sum += Q(i,j)*m_ktrtr(i,j)*m_scale*2.0;
	}

	sum /= 2.0;

	SGVector<float64_t> scale(1);

	scale[0] = sum;

	gradient.add(param, scale);
	para_dict.add(param, this);

	index = m_model->get_modsel_param_index("sigma");
	param = m_model->m_model_selection_parameters->get_parameter(index);

	sum = m_sigma*Q.trace(Q.matrix, Q.num_rows, Q.num_cols);

	SGVector<float64_t> sigma(1);

	sigma[0] = sum;
	gradient.add(param, sigma);
	para_dict.add(param, m_model);

	return gradient;

}

SGVector<float64_t> CExactInferenceMethod::get_diagonal_vector()
{
	if(update_parameter_hash())
		update_all();

	check_members();

	float64_t m_sigma =
			dynamic_cast<CGaussianLikelihood*>(m_model)->get_sigma();

	SGVector<float64_t> result =
			SGVector<float64_t>(m_features->get_num_vectors());

	result.fill_vector(result.vector, m_features->get_num_vectors(),
			1.0/m_sigma);

	return result;
}

float64_t CExactInferenceMethod::get_negative_marginal_likelihood()
{
	if(update_parameter_hash())
		update_all();

	float64_t result;

	result = m_label_vector.dot(m_label_vector.vector, m_alpha.vector,
			m_label_vector.vlen)/2.0;

	float64_t m_sigma =
			dynamic_cast<CGaussianLikelihood*>(m_model)->get_sigma();

	for (int i = 0; i < m_L.num_rows; i++)
		result += CMath::log(m_L(i,i));

	result += m_L.num_rows * CMath::log(2*CMath::PI*m_sigma*m_sigma)/2.0;

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

	float64_t sigma =
			dynamic_cast<CGaussianLikelihood*>(m_model)->get_sigma();

	/* temporaryly add noise */
	float64_t noise=(m_scale*m_scale)/(sigma*sigma);
	for (index_t i=0; i<m_ktrtr.num_rows; ++i)
		m_ktrtr(i, i)+=noise;

	/* check whether to allocate cholesky mempory */
	if (!m_L.matrix || m_L.num_rows!=m_ktrtr.num_rows)
		m_L=SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);

	/* creates views on kernel and cholesky matrix and perform cholesky */
	Map<MatrixXd> K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> L(m_L.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	LLT<MatrixXd> llt;
	llt.compute(K);
	L=llt.matrixU();
	cout << L << endl;

	/* remove noise again */
	for (index_t i=0; i<m_ktrtr.num_rows; ++i)
		m_ktrtr(i, i)-=noise;
}

void CExactInferenceMethod::update_alpha()
{
	float64_t sigma =
			dynamic_cast<CGaussianLikelihood*>(m_model)->get_sigma();

	for (int i = 0; i < m_label_vector.vlen; i++)
		m_label_vector[i] = m_label_vector[i] - m_data_means[i];

	m_alpha = SGVector<float64_t>(m_label_vector.vlen);

	/* temporaryly add noise */
	float64_t noise=(m_scale*m_scale)/(sigma*sigma);
	for (index_t i=0; i<m_ktrtr.num_rows; ++i)
		m_ktrtr(i, i)+=noise;

	/* creates views on kernel matrix, labels, and alpha and solve system
	 * (K(X, X)+sigma*I) a = y for a */
	Map<MatrixXd> K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> a(m_alpha.vector, m_alpha.vlen);
	Map<VectorXd> y(m_label_vector.vector, m_label_vector.vlen);

	LLT<MatrixXd> llt;
	llt.compute(K);
	a=llt.solve(y);

	/* remove noise again */
	for (index_t i=0; i<m_ktrtr.num_rows; ++i)
		m_ktrtr(i, i)-=noise;

}
#endif
#endif // HAVE_LAPACK
