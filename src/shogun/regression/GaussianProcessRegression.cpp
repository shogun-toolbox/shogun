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

using namespace shogun;

CGaussianProcessRegression::CGaussianProcessRegression()
: CMachine()
{
	init();

}

CGaussianProcessRegression::CGaussianProcessRegression(float64_t sigma, 
CKernel* k, CSimpleFeatures<float64_t>* data, CLabels* lab)
: CMachine()
{
	init();

	m_sigma = sigma;
	
	set_labels(lab);
	set_features(data);
	set_kernel(k);
}

void CGaussianProcessRegression::init()
{
	m_sigma = 1.0;
	kernel = NULL;
	features = NULL;
	
	SG_ADD((CSGObject**) &kernel, "kernel", "", MS_AVAILABLE);
	SG_ADD((CSGObject**) &features, "features", "Feature object.",
	    MS_NOT_AVAILABLE);
	SG_ADD(&m_sigma, "sigma", "Sigma.", MS_AVAILABLE);
}

CLabels* CGaussianProcessRegression::mean_prediction(CFeatures* data)
{
	if (!kernel)
		SG_ERROR( "No kernel assigned!\n");
	if (m_labels->get_num_labels() != features->get_num_vectors())
		SG_ERROR("Number of training vectors does not match number of labels\n");
	if (m_labels->get_num_labels() != m_alpha.vlen)
		SG_ERROR("Machine not properly trained.\n");

	kernel->init(data, features);
	
	//K(X_test, X_train)
	SGMatrix<float64_t> kernel_test_matrix = kernel->get_kernel_matrix();
			
	SGVector< float64_t > result_vector(m_labels->get_num_labels());
	
	//Here we multiply K*^t by alpha to receive the mean predictions.
	cblas_dgemv(CblasColMajor, CblasTrans, kernel_test_matrix.num_rows, 
		    m_alpha.vlen, 1.0, kernel_test_matrix.matrix, 
		    kernel_test_matrix.num_cols, m_alpha.vector, 1, 0.0, 
		    result_vector.vector, 1);
	
	CLabels* result = new CLabels(result_vector);
	
	kernel_test_matrix.destroy_matrix();
	
	return result;
}

CLabels* CGaussianProcessRegression::apply()
{
	if (!features)
		return NULL;
	
	return mean_prediction(features);
}

CLabels* CGaussianProcessRegression::apply(CFeatures* data)
{
	if (!data)
		SG_ERROR("No features specified\n");
	if (!data->has_property(FP_DOT))
		SG_ERROR("Specified features are not of type CDotFeatures\n");
	if (m_labels->get_num_labels() != features->get_num_vectors())
		SG_ERROR("Number of training vectors does not match number of labels\n");
	if (data->get_feature_class() != C_SIMPLE)
		SG_ERROR("Expected Simple Features\n");
	if (data->get_feature_type() != F_DREAL)
		SG_ERROR("Expected Real Features\n");
	if (!kernel)
		SG_ERROR( "No kernel assigned!\n");

	return mean_prediction(data);
}

float64_t CGaussianProcessRegression::apply(int32_t num)
{
	SG_ERROR("apply(int32_t num) is not yet implemented "
	"for %s\n", get_name());
	
	return 0;
}


bool CGaussianProcessRegression::train_machine(CFeatures* data)
{
  	if (!data->has_property(FP_DOT))
		SG_ERROR("Specified features are not of type CDotFeatures\n");
  	if (m_labels->get_num_labels() != data->get_num_vectors())
		SG_ERROR("Number of training vectors does not match number of labels\n");
	if (data->get_feature_class() != C_SIMPLE)
		SG_ERROR("Expected Simple Features\n");
	if (data->get_feature_type() != F_DREAL)
		SG_ERROR("Expected Real Features\n");
	if (!kernel)
		SG_ERROR( "No kernel assigned!\n");
	
	set_features((CDotFeatures*)data);
	
	kernel->init(features, features);
	
	//K(X_train, X_train)
	SGMatrix<float64_t> kernel_train_matrix = kernel->get_kernel_matrix();
	
	SGMatrix<float64_t> temp1(kernel_train_matrix.num_rows,
	kernel_train_matrix.num_cols);
	
	SGMatrix<float64_t> temp2(kernel_train_matrix.num_rows,
	kernel_train_matrix.num_cols);
	
	SGVector<float64_t> diagonal(temp1.num_rows);
	CMath::fill_vector(diagonal.vector, temp1.num_rows, 1.0);
	
	CMath::create_diagonal_matrix(temp1.matrix, diagonal.vector, temp1.num_rows);
	CMath::create_diagonal_matrix(temp2.matrix, diagonal.vector, temp2.num_rows);
		
	//Calculate first (K(X_train, X_train)+sigma*I)
	cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, 
		kernel_train_matrix.num_rows, temp2.num_cols, 1.0, 
		kernel_train_matrix.matrix, kernel_train_matrix.num_cols,
		temp2.matrix, temp2.num_cols, m_sigma*m_sigma, 
		temp1.matrix, temp1.num_cols);
		
	memcpy(temp2.matrix, temp1.matrix, 
		temp2.num_cols*temp2.num_rows*sizeof(float64_t));
	
	//Get Lower triangle cholesky decomposition of K(X_train, X_train)+sigma*I)
	clapack_dpotrf(CblasColMajor, CblasLower, 
		temp1.num_cols, temp1.matrix, temp1.num_cols);
	
	m_L.destroy_matrix();
	
	m_L = SGMatrix<float64_t>(temp1.num_rows, temp1.num_cols);
	memcpy(m_L.matrix, temp1.matrix,
		temp1.num_cols*temp1.num_rows*sizeof(float64_t));
		
	SGVector<float64_t> label_vector = m_labels->get_labels();
	
	m_alpha.destroy_vector();
	m_alpha = SGVector<float64_t>(label_vector.vlen);
	memcpy(m_alpha.vector, label_vector.vector,
		label_vector.vlen*sizeof(float64_t));
	
	//Solve (K(X_train, X_train)+sigma*I) alpha = labels for alpha.
	clapack_dposv(CblasColMajor, CblasLower,
		  temp2.num_cols, 1, temp2.matrix, temp2.num_cols,
		  m_alpha.vector, temp2.num_cols);
	
	temp1.destroy_matrix();
	temp2.destroy_matrix();
	kernel_train_matrix.destroy_matrix();
	diagonal.destroy_vector();
	
	return true;
}

SGMatrix<float64_t> CGaussianProcessRegression::getCovarianceMatrix(CFeatures* data)
{
	if (m_labels->get_num_labels() != features->get_num_vectors())
		SG_ERROR("Number of training vectors does not match number of labels.\n");
	if (!kernel)
		SG_ERROR( "No kernel assigned!\n");
	if (features->get_num_vectors() != m_L.num_rows)
		SG_ERROR("Machine not properly trained.\n");

	kernel->init(data, features);
	
	//K(X_test, X_train)
	SGMatrix<float64_t> kernel_test_matrix = kernel->get_kernel_matrix();
	
	kernel->init(data, data);
	
	//K(X_test, X_test)
	SGMatrix<float64_t> kernel_star_matrix = kernel->get_kernel_matrix();
	
	SGMatrix<float64_t> temp1(kernel_test_matrix.num_rows,
		    kernel_test_matrix.num_cols);

	SGMatrix<float64_t> temp2(kernel_test_matrix.num_rows,
		    kernel_test_matrix.num_cols);
		
	SGMatrix<float64_t> temp3(kernel_test_matrix.num_rows,
		    kernel_test_matrix.num_cols);
	
	//Indices used to solve Lv=K(X_test, X_train) for v
	SGVector< int32_t > ipiv(CMath::min(m_L.num_rows, m_L.num_cols));
	
	int info;

	
	memcpy(temp1.matrix, kernel_test_matrix.matrix, 
		kernel_test_matrix.num_cols*kernel_test_matrix.num_rows*sizeof(float64_t));
	
	//Get indices used to solve Lv=K(X_test, X_train) for v
	dgetrf_(&m_L.num_rows, &m_L.num_cols, m_L.matrix, &m_L.num_cols, ipiv.vector, &info);
	
	//Solve Lv=K(X_test, X_train) for v
	clapack_dgetrs(CblasColMajor, CblasNoTrans,
		    m_L.num_rows, kernel_test_matrix.num_cols, m_L.matrix, m_L.num_cols,
		    ipiv.vector, temp1.matrix, temp1.num_cols);
	
	memcpy(temp2.matrix, temp1.matrix, temp1.num_cols*temp1.num_rows*sizeof(float64_t));
	
	//Store v^t*v in temp3
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, temp1.num_rows, 
		    temp1.num_cols, temp1.num_cols, 1.0, temp1.matrix, temp1.num_cols, 
		    temp2.matrix, temp2.num_cols, 0.0, temp3.matrix, temp3.num_cols);

	//Set temp2 to identity matrix
	SGVector<float64_t> diagonal(temp2.num_rows);
	CMath::fill_vector(diagonal.vector, temp2.num_rows, 1.0);
	CMath::create_diagonal_matrix(temp2.matrix, diagonal.vector, temp2.num_rows);
	
	//Calculate Covariance Matrix = K(X_test, X_test) - v^t*v
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, kernel_star_matrix.num_rows, 
		    kernel_star_matrix.num_cols, kernel_star_matrix.num_cols, 1.0,
		    kernel_star_matrix.matrix, kernel_star_matrix.num_cols, 
		    temp2.matrix, temp2.num_cols, -1.0, temp3.matrix, temp3.num_cols);
	
	temp1.destroy_matrix();
	temp2.destroy_matrix();
	ipiv.destroy_vector();
	kernel_star_matrix.destroy_matrix();
	kernel_test_matrix.destroy_matrix();
	diagonal.destroy_vector();
	
	return temp3;
}


CGaussianProcessRegression::~CGaussianProcessRegression()
{
	SG_UNREF(kernel);
	SG_UNREF(features);
	m_L.destroy_matrix();
	m_alpha.destroy_vector();
}

void CGaussianProcessRegression::set_kernel(CKernel* k)
{
	SG_REF(k);
	SG_UNREF(kernel);
	kernel=k;
}

bool CGaussianProcessRegression::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

CKernel* CGaussianProcessRegression::get_kernel()
{
	SG_REF(kernel);
	return kernel;
}

bool CGaussianProcessRegression::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}
#endif
