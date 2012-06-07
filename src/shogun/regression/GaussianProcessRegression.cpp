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
		if (m_labels->get_num_labels() != features->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n");
		if (data->get_feature_class() != C_DENSE)
			SG_ERROR("Expected Simple Features\n");
		if (data->get_feature_type() != F_DREAL)
			SG_ERROR("Expected Real Features\n");

		set_features((CDotFeatures*)data);
	}

	SGVector<float64_t> m_alpha = m_method->get_alpha();
	CKernel* kernel = m_method->get_kernel();

	kernel->init(data, features);
	
	//K(X_test, X_train)
	SGMatrix<float64_t> kernel_test_matrix = kernel->get_kernel_matrix();
			
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

SGMatrix<float64_t> CGaussianProcessRegression::getCovarianceMatrix(CFeatures* data)
{
	if (m_labels->get_num_labels() != features->get_num_vectors())
		SG_ERROR("Number of training vectors does not match number of labels.\n");


	SGVector<float64_t> diagonal = m_method->get_diagonal_vector();
	CKernel* kernel = m_method->get_kernel();
	
	/*do some matrix algebra here to recover posterior covariance*/
	
	return SGMatrix<float64_t>(0,0);
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
