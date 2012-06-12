/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
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

	features = NULL;
	m_method = NULL;
	
	SG_ADD((CSGObject**) &features, "features", "Feature object.",
	    MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &m_method, "m_method", "Inference Method.",
	    MS_NOT_AVAILABLE);
}

CRegressionLabels* CGaussianProcessRegression::mean_prediction(CFeatures* data)
{
	return NULL;
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
	}

	SGVector<float64_t> m_alpha = m_method->get_alpha();
	CKernel* kernel = m_method->get_kernel();

	kernel->init(features, data);
	
	//K(X_test, X_train)
	SGMatrix<float64_t> kernel_test_matrix = kernel->get_kernel_matrix();

	//CMath::display_matrix(kernel_test_matrix.matrix, kernel_test_matrix.num_rows, kernel_test_matrix.num_cols);
	SGVector< float64_t > result_vector(m_labels->get_num_labels());
	
	//Here we multiply K*^t by alpha to receive the mean predictions.
	cblas_dgemv(CblasColMajor, CblasTrans, kernel_test_matrix.num_rows,
		    m_alpha.vlen, 1.0, kernel_test_matrix.matrix, 
		    kernel_test_matrix.num_cols, m_alpha.vector, 1, 0.0, 
		    result_vector.vector, 1);
	
	CRegressionLabels* result = new CRegressionLabels(result_vector);
	
	return result;
}

bool CGaussianProcessRegression::train_machine(CFeatures* data)
{
	return false;
}

SGVector<float64_t> CGaussianProcessRegression::getCovarianceVector(CFeatures* data)
{
	if (m_labels->get_num_labels() != features->get_num_vectors())
		SG_ERROR("Number of training vectors does not match number of labels.\n");


	SGVector<float64_t> diagonal = m_method->get_diagonal_vector();
	SGVector<float64_t> diagonal2(data->get_num_vectors());
	
	SGMatrix<float64_t> temp1(diagonal.vlen, data->get_num_vectors());

	SGMatrix<float64_t> m_L = m_method->get_cholesky();

	SGMatrix<float64_t> temp2(m_L.num_rows, m_L.num_cols);

	CKernel* kernel = m_method->get_kernel();

	kernel->init(features, data);

	//K(X_test, X_train)
	SGMatrix<float64_t> kernel_test_matrix = kernel->get_kernel_matrix();


	for(int i = 0; i < diagonal.vlen; i++)
	{
		for(int j = 0; j < data->get_num_vectors(); j++)
		{
			temp1(j,i) = diagonal[i]*kernel_test_matrix(j,i);
		}
	}

	for(int i = 0; i < diagonal2.vlen; i++)
	{
		diagonal2[i] = 0;
	}

	memcpy(temp2.matrix, m_L.matrix,
			m_L.num_cols*m_L.num_rows*sizeof(float64_t));

	CMath::transpose_matrix(temp2.matrix, temp2.num_rows, temp2.num_cols);

	clapack_dpotrs(CblasColMajor, CblasUpper,
			temp2.num_rows, temp1.num_cols, temp2.matrix, temp1.num_cols,
		  temp2.matrix, temp2.num_cols);

	CMath::display_matrix(temp2.matrix, temp2.num_rows, temp2.num_cols);


	for(int i = 0; i < temp2.num_rows; i++)
	{
		for(int j = 0; j < temp2.num_cols; j++)
		{
			temp2(i,j) = temp2(i,j)*temp2(i,j);
		}
	}

	for(int i = 0; i < temp2.num_cols; i++)
	{
		diagonal2[i] = 0;

		for(int j = 0; j < temp2.num_rows; j++)
		{
			diagonal2[i] += temp2(j,i);
		}
	}

	kernel->init(data, data);

	//K(X_test, X_train)
	SGMatrix<float64_t> kernel_test_matrix2 = kernel->get_kernel_matrix();

	SGVector<float64_t> result(kernel_test_matrix2.num_cols);

	for(int i = 0; i < kernel_test_matrix2.num_cols; i++)
	{
		kernel_test_matrix2(i,i) -= diagonal2[i];
		result[i] = kernel_test_matrix2(i,i);
	}

	return result;
}


CGaussianProcessRegression::~CGaussianProcessRegression()
{
	SG_UNREF(features);
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
