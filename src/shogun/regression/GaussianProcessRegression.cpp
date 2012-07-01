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

#include <shogun/io/SGIO.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/labels/RegressionLabels.h>

using namespace shogun;

CGaussianProcessRegression::CGaussianProcessRegression()
: CMachine()
{
	init();
}

CGaussianProcessRegression::CGaussianProcessRegression(CInferenceMethod* inf,
		   CDenseFeatures<float64_t>* data, CLabels* lab)
: CMachine()
{
	init();
	
	set_labels(lab);
	set_features(data);
	set_method(inf);
}

void CGaussianProcessRegression::init()
{

	m_features = NULL;
	m_method = NULL;
	m_data = NULL;
	m_return = GP_RETURN_MEANS;

	SG_ADD((CSGObject**) &m_features, "features", "Feature object.",
	    MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &m_method, "inference_method", "Inference Method.",
	    MS_AVAILABLE);
}

void CGaussianProcessRegression::update_kernel_matrices()
{
	CKernel* kernel = NULL;

	if (m_method)
		kernel = m_method->get_kernel();

	if (kernel)
	{
		float64_t m_scale = m_method->get_scale();

		kernel->cleanup();

		kernel->init(m_features, m_data);

		//K(X_test, X_train)
		m_k_trts = kernel->get_kernel_matrix();

		for (int i = 0; i < m_k_trts.num_rows; i++)
		{
			for (int j = 0; j < m_k_trts.num_cols; j++)
				m_k_trts(i,j) *= (m_scale*m_scale);
		}

		kernel->cleanup();

		kernel->init(m_data, m_data);

		m_k_tsts = kernel->get_kernel_matrix();

		for (int i = 0; i < m_k_tsts.num_rows; i++)
		{
			for (int j = 0; j < m_k_tsts.num_cols; j++)
				m_k_tsts(i,j) *= (m_scale*m_scale);
		}

		SG_UNREF(kernel);
	}
}

CRegressionLabels* CGaussianProcessRegression::apply_regression(CFeatures* data)
{

	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n");
		if (data->get_feature_class() != C_DENSE)
			SG_ERROR("Expected Simple Features\n");
		if (data->get_feature_type() != F_DREAL)
			SG_ERROR("Expected Real Features\n");

			SG_UNREF(m_data);
			SG_REF(data);
			m_data = (CDotFeatures*)data;
			update_kernel_matrices();
	}

	else if (!m_data)
		SG_ERROR("No testing features!\n");

	else if (update_parameter_hash())
		update_kernel_matrices();

	if (m_return == GP_RETURN_COV)
	{
		CRegressionLabels* result =
				new CRegressionLabels(getCovarianceVector());

		return result;
	}

	if (m_return == GP_RETURN_MEANS)
	{
		CRegressionLabels* result =
				new CRegressionLabels(getMeanVector());

		return result;
	}

	else
	{

		SGVector<float64_t> mean_vector = getMeanVector();
		SGVector<float64_t> cov_vector = getCovarianceVector();

		int size = mean_vector.vlen+cov_vector.vlen;

		SGVector<float64_t> result_vector(size);

		for (int i = 0; i < size; i++)
		{
			if (i < mean_vector.vlen)
				result_vector[i] = mean_vector[i];
			else
				result_vector[i] = cov_vector[i-mean_vector.vlen];
		}

		CRegressionLabels* result =
				new CRegressionLabels(result_vector);

		return result;
	}

}

bool CGaussianProcessRegression::train_machine(CFeatures* data)
{
	return false;
}


SGVector<float64_t> CGaussianProcessRegression::getMeanVector()
{

	SGVector<float64_t> m_alpha = m_method->get_alpha();

	SGVector< float64_t > result_vector(m_labels->get_num_labels());
	
	//Here we multiply K*^t by alpha to receive the mean predictions.
	cblas_dgemv(CblasColMajor, CblasTrans, m_k_trts.num_rows,
		    m_alpha.vlen, 1.0, m_k_trts.matrix,
		    m_k_trts.num_cols, m_alpha.vector, 1, 0.0,
		    result_vector.vector, 1);
	
	CLikelihoodModel* lik = m_method->get_model();

	result_vector = lik->evaluate_means(result_vector);

	SG_UNREF(lik);

	return result_vector;
}


SGVector<float64_t> CGaussianProcessRegression::getCovarianceVector()
{

	if (!m_data)
		SG_ERROR("No testing features!\n");

	SGVector<float64_t> diagonal = m_method->get_diagonal_vector();
	SGVector<float64_t> diagonal2(m_data->get_num_vectors());

	SGMatrix<float64_t> temp1(m_data->get_num_vectors(), diagonal.vlen);

	SGMatrix<float64_t> m_L = m_method->get_cholesky();

	SGMatrix<float64_t> temp2(m_L.num_rows, m_L.num_cols);

	for (int i = 0; i < diagonal.vlen; i++)
	{
		for (int j = 0; j < m_data->get_num_vectors(); j++)
			temp1(j,i) = diagonal[i]*m_k_trts(j,i);
	}

	for (int i = 0; i < diagonal2.vlen; i++)
		diagonal2[i] = 0;

	memcpy(temp2.matrix, m_L.matrix,
			m_L.num_cols*m_L.num_rows*sizeof(float64_t));


	temp2.transpose_matrix(temp2.matrix, temp2.num_rows, temp2.num_cols);

	SGVector<int32_t> ipiv(temp2.num_cols);

	//Solve L^T V = K(Train,Test)*Diagonal Vector Entries for V
	clapack_dgetrf(CblasColMajor, temp2.num_rows, temp2.num_cols,
			temp2.matrix, temp2.num_cols, ipiv.vector);

	clapack_dgetrs(CblasColMajor, CblasNoTrans,
	                   temp2.num_rows, temp1.num_cols, temp2.matrix,
	                   temp2.num_cols, ipiv.vector, temp1.matrix,
	                   temp1.num_cols);

	for (int i = 0; i < temp1.num_rows; i++)
	{
		for (int j = 0; j < temp1.num_cols; j++)
			temp1(i,j) = temp1(i,j)*temp1(i,j);
	}

	for (int i = 0; i < temp1.num_cols; i++)
	{
		diagonal2[i] = 0;

		for (int j = 0; j < temp1.num_rows; j++)
			diagonal2[i] += temp1(j,i);
	}


	SGVector<float64_t> result(m_k_tsts.num_cols);

	//Subtract V from K(Test,Test) to get covariances.
	for (int i = 0; i < m_k_tsts.num_cols; i++)
		result[i] = m_k_tsts(i,i) - diagonal2[i];

	CLikelihoodModel* lik = m_method->get_model();

	SG_UNREF(lik);

	return lik->evaluate_variances(result);
}


CGaussianProcessRegression::~CGaussianProcessRegression()
{
	SG_UNREF(m_features);
	SG_UNREF(m_method);
	SG_UNREF(m_data);
}

void CGaussianProcessRegression::set_kernel(CKernel* k)
{
	m_method->set_kernel(k);
}

bool CGaussianProcessRegression::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

CKernel* CGaussianProcessRegression::get_kernel()
{
	return m_method->get_kernel();
}

bool CGaussianProcessRegression::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}
#endif
