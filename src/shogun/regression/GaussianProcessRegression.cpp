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
	m_return = GP_RETURN_MEANS;

	SG_ADD((CSGObject**) &m_features, "features", "Feature object.",
	    MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &m_method, "inference_method", "Inference Method.",
	    MS_AVAILABLE);
}

CRegressionLabels* CGaussianProcessRegression::apply_regression(CFeatures* data)
{
	if (m_return == GP_RETURN_COV)
	{
		CRegressionLabels* result =
				new CRegressionLabels(getCovarianceVector(data));

		return result;
	}

	if (m_return == GP_RETURN_MEANS)
	{
		CRegressionLabels* result =
				new CRegressionLabels(getMeanVector(data));

		return result;
	}

	else
	{
		SG_REF(data);
		SGVector<float64_t> mean_vector = getMeanVector(data);
		SGVector<float64_t> cov_vector = getCovarianceVector(data);

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

		SG_UNREF(data);

		return result;
	}

}

bool CGaussianProcessRegression::train_machine(CFeatures* data)
{
	return false;
}


SGVector<float64_t> CGaussianProcessRegression::getMeanVector(
		CFeatures* data)
{
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n");
		if (data->get_feature_class() != C_DENSE)
			SG_ERROR("Expected Simple Features\n");
		if (data->get_feature_type() != F_DREAL)
			SG_ERROR("Expected Real Features\n");
	}

	else
		SG_ERROR("Null data vector!\n");

	SGVector<float64_t> m_alpha = m_method->get_alpha();
	float64_t m_scale = m_method->get_scale();
	CKernel* kernel = m_method->get_kernel();

	kernel->cleanup();
	
	kernel->init(m_features, data);
	
	//K(X_test, X_train)
	SGMatrix<float64_t> kernel_test_matrix = kernel->get_kernel_matrix();

	for (int i = 0; i < kernel_test_matrix.num_rows; i++)
	{
		for (int j = 0; j < kernel_test_matrix.num_cols; j++)
			kernel_test_matrix(i,j) *= (m_scale*m_scale);
	}

	SGVector< float64_t > result_vector(m_labels->get_num_labels());
	
	//Here we multiply K*^t by alpha to receive the mean predictions.
	cblas_dgemv(CblasColMajor, CblasTrans, kernel_test_matrix.num_rows,
		    m_alpha.vlen, 1.0, kernel_test_matrix.matrix, 
		    kernel_test_matrix.num_cols, m_alpha.vector, 1, 0.0, 
		    result_vector.vector, 1);
	
	CLikelihoodModel* lik = m_method->get_model();

	result_vector = lik->evaluate_means(result_vector);

	
	SG_UNREF(kernel);
	SG_UNREF(lik);
//	SG_REF(result);

	return result_vector;
}


SGVector<float64_t> CGaussianProcessRegression::getCovarianceVector(
		CFeatures* data)
{
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n");
		if (data->get_feature_class() != C_DENSE)
			SG_ERROR("Expected Simple Features\n");
		if (data->get_feature_type() != F_DREAL)
			SG_ERROR("Expected Real Features\n");
	}

	else
		SG_ERROR("Null data vector!\n");

	SG_REF(data);
	SGVector<float64_t> diagonal = m_method->get_diagonal_vector();
	SGVector<float64_t> diagonal2(data->get_num_vectors());

	SGMatrix<float64_t> temp1(data->get_num_vectors(), diagonal.vlen);

	SGMatrix<float64_t> m_L = m_method->get_cholesky();

	SGMatrix<float64_t> temp2(m_L.num_rows, m_L.num_cols);

	CKernel* kernel = m_method->get_kernel();

	float64_t m_scale = m_method->get_scale();

	kernel->cleanup();

	kernel->init(m_features, data);

	//K(X_test, X_train)
	SGMatrix<float64_t> kernel_test_matrix = kernel->get_kernel_matrix();

	for (int i = 0; i < kernel_test_matrix.num_rows; i++)
	{
		for (int j = 0; j < kernel_test_matrix.num_cols; j++)
			kernel_test_matrix(i,j) *= (m_scale*m_scale);
	}

	for (int i = 0; i < diagonal.vlen; i++)
	{
		for (int j = 0; j < data->get_num_vectors(); j++)
			temp1(j,i) = diagonal[i]*kernel_test_matrix(j,i);
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

	kernel->cleanup();

	kernel->init(data, data);

	//K(X_test, X_test)

	SGMatrix<float64_t> kernel_test_matrix2 = kernel->get_kernel_matrix();

	for (int i = 0; i < kernel_test_matrix2.num_rows; i++)
	{
		for (int j = 0; j < kernel_test_matrix2.num_cols; j++)
			kernel_test_matrix2(i,j) *= (m_scale*m_scale);
	}

	SGVector<float64_t> result(kernel_test_matrix2.num_cols);

	//Subtract V from K(Test,Test) to get covariances.
	for (int i = 0; i < kernel_test_matrix2.num_cols; i++)
	{
		kernel_test_matrix2(i,i) -= diagonal2[i];
		result[i] = kernel_test_matrix2(i,i);
	}

	CLikelihoodModel* lik = m_method->get_model();

	SG_UNREF(data);
	SG_UNREF(kernel);
	SG_UNREF(lik);

	return lik->evaluate_variances(result);
}


CGaussianProcessRegression::~CGaussianProcessRegression()
{
	SG_UNREF(m_features);
	SG_UNREF(m_method);
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
