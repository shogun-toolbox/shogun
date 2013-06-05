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

#include <shogun/io/SGIO.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/regression/gp/FITCInferenceMethod.h>

using namespace shogun;

#include <Eigen/Dense>

using namespace Eigen;

CGaussianProcessRegression::CGaussianProcessRegression()
: CMachine()
{
	init();
}

CGaussianProcessRegression::CGaussianProcessRegression(CInferenceMethod* inf, CFeatures* data, CLabels* lab)
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

		/* set training data to latent features if exist, otherwise
		 * training features */
		if (m_method->get_inference_type()==INF_FITC)
		{
			CFITCInferenceMethod* inf=CFITCInferenceMethod::obtain_from_generic(m_method);
			CFeatures* latent_features=inf->get_latent_features();
			kernel->init(latent_features, m_data);
			SG_UNREF(inf);
			SG_UNREF(latent_features);
		}
		else
		{
			CFeatures* features=m_method->get_features();
			kernel->init(features, m_data);
			SG_UNREF(features);
		}

		//K(X_train, X_test)
		m_k_trts = kernel->get_kernel_matrix();

		for (index_t i = 0; i < m_k_trts.num_rows; i++)
		{
			for (index_t j = 0; j < m_k_trts.num_cols; j++)
				m_k_trts(i,j) *= (m_scale*m_scale);
		}

		kernel->init(m_data, m_data);
		m_k_tsts = kernel->get_kernel_matrix();

		for (index_t i = 0; i < m_k_tsts.num_rows; i++)
		{
			for (index_t j = 0; j < m_k_tsts.num_cols; j++)
				m_k_tsts(i,j) *= (m_scale*m_scale);
		}

		kernel->remove_lhs_and_rhs();

		SG_UNREF(kernel);
	}
}

CRegressionLabels* CGaussianProcessRegression::apply_regression(CFeatures* data)
{

	if (data)
	{
		if (data->get_feature_class() == C_COMBINED)
		{
			SG_WARNING("%s::apply_regression(): This only works for combined"
						" features which all share the same underlying object!\n",
						get_name());

			CDotFeatures* feat =
					(CDotFeatures*)((CCombinedFeatures*)data)->
					get_first_feature_obj();

			if (!feat->has_property(FP_DOT))
				SG_ERROR("Specified features are not of type CFeatures\n")

			if (feat->get_feature_class() != C_DENSE)
				SG_ERROR("Expected Dense Features\n")

			if (feat->get_feature_type() != F_DREAL)
				SG_ERROR("Expected Real Features\n")

			SG_UNREF(feat);
		}

		else
		{
			if (!data->has_property(FP_DOT))
				SG_ERROR("Specified features are not of type CFeatures\n")

			if (data->get_feature_class() != C_DENSE)
				SG_ERROR("Expected Simple Features\n")

			if (data->get_feature_type() != F_DREAL)
				SG_ERROR("Expected Real Features\n")
		}

		SG_REF(data);
		SG_UNREF(m_data);
		m_data = (CFeatures*)data;
		update_kernel_matrices();
	}

	else if (!m_data)
		SG_ERROR("No testing features!\n")

	else if (update_parameter_hash())
		update_kernel_matrices();

	if (m_return == GP_RETURN_COV)
	{
		CRegressionLabels* result =
				new CRegressionLabels(get_covariance_vector());

		return result;
	}

	if (m_return == GP_RETURN_MEANS)
	{
		CRegressionLabels* result =
				new CRegressionLabels(get_mean_vector());

		return result;
	}

	else
	{

		SGVector<float64_t> mean_vector = get_mean_vector();
		SGVector<float64_t> cov_vector = get_covariance_vector();

		index_t size = mean_vector.vlen+cov_vector.vlen;

		SGVector<float64_t> result_vector(size);

		for (index_t i = 0; i < size; i++)
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


SGVector<float64_t> CGaussianProcessRegression::get_mean_vector()
{

	SGVector<float64_t> alpha = m_method->get_alpha();

	CMeanFunction* mean_function = m_method->get_mean();

	SGMatrix<float64_t> features;
	if(m_data->get_feature_class() == C_COMBINED)
	{
		SG_WARNING("%s::get_mean_vector(): This only works for combined"
				" features which all share the same underlying object!\n",
				get_name());
		features = ((CDotFeatures*)((CCombinedFeatures*)m_data)->
									get_first_feature_obj())->
											get_computed_dot_feature_matrix();
	}

	else
	{
		features = ((CDotFeatures*)m_data)->
			get_computed_dot_feature_matrix();
	}

	REQUIRE(mean_function, "%s::get_mean_vector(): Mean function is NULL!\n",
			get_name());

	SGVector<float64_t> means = mean_function->get_mean_vector(features);
	SGVector< float64_t > result_vector(features.num_cols);

	/* create eigen3 views an multiply */
	Map<MatrixXd> K(m_k_trts.matrix, m_k_trts.num_rows, m_k_trts.num_cols);
	Map<VectorXd> a(alpha.vector, alpha.vlen);
	Map<VectorXd> r(result_vector.vector, result_vector.vlen);
	r=K.transpose()*a;

	for (index_t i = 0; i < result_vector.vlen; i++)
		result_vector[i] += means[i];

	CLikelihoodModel* lik = m_method->get_model();

	result_vector = lik->evaluate_means(result_vector);

	SG_UNREF(lik);
	SG_UNREF(mean_function);

	return result_vector;
}

SGVector<float64_t> CGaussianProcessRegression::get_covariance_vector()
{
	if (!m_data)
		SG_ERROR("No testing features!\n")

	// get shogun of diagonal sigma vector and create eigen representation
	SGVector<float64_t> diagonal=m_method->get_diagonal_vector();
	Map<VectorXd> eigen_diagonal(diagonal.vector, diagonal.vlen);

	// get shogun representation of cholesky and create eigen representation
	SGMatrix<float64_t> L=m_method->get_cholesky();
	Map<MatrixXd> eigen_L(L.matrix, L.num_rows, L.num_cols);

	// create eigen representation of K(train, tests) kernel matrix
	Map<MatrixXd> eigen_Ks(m_k_trts.matrix, m_k_trts.num_rows, m_k_trts.num_cols);

	// variance vector
	SGVector<float64_t> s2(m_k_tsts.num_cols);

	if (diagonal.vlen>0)
	{
		// solve L' * V = diagonal * Ks and compute V.^2
		MatrixXd eigen_V=eigen_L.triangularView<Upper>().adjoint().solve(
			eigen_diagonal.asDiagonal()*eigen_Ks);
		MatrixXd eigen_sV=eigen_V.cwiseProduct(eigen_V);

		VectorXd eigen_diagonal2=eigen_sV.colwise().sum();

		for (index_t i=0; i<m_k_tsts.num_cols; i++)
			s2[i]=m_k_tsts(i,i)-eigen_diagonal2[i];
	}
	else
	{
		// M = Ks .* (L * Ks)
		MatrixXd eigen_M=eigen_Ks.cwiseProduct(eigen_L*eigen_Ks);

		VectorXd eigen_diagonal2=eigen_M.colwise().sum();

		for (index_t i=0; i<m_k_tsts.num_cols; i++)
			s2[i]=m_k_tsts(i,i)+eigen_diagonal2[i];
	}

	CLikelihoodModel* lik=m_method->get_model();
	s2=lik->evaluate_variances(s2);
	SG_UNREF(lik);

	return s2;
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
