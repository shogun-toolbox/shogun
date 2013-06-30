/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 *
 * Code adapted from Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */

#include <shogun/regression/GaussianProcessRegression.h>

#ifdef HAVE_EIGEN3

#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/gp/FITCInferenceMethod.h>

#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CGaussianProcessRegression::CGaussianProcessRegression() : CGaussianProcessMachine()
{
}

CGaussianProcessRegression::CGaussianProcessRegression(CInferenceMethod* method)
	: CGaussianProcessMachine(method)
{
	// set labels
	m_labels=method->get_labels();
}

CGaussianProcessRegression::~CGaussianProcessRegression()
{
}

CRegressionLabels* CGaussianProcessRegression::apply_regression(CFeatures* data)
{
	// check whether given combination of inference method and likelihood function
	// supports regression
	REQUIRE(m_method, "Inference method must be attached\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_regression(), "%s with %s doesn't support regression\n",
			m_method->get_name(), lik->get_name())
	SG_UNREF(lik);

	CRegressionLabels* result;

	// if regression data equals to NULL, then apply regression on training features
	if (!data)
	{
		CFeatures* feat;

		// use latent features for FITC inference method
		if (m_method->get_inference_type()==INF_FITC)
		{
			CFITCInferenceMethod* fitc_method=CFITCInferenceMethod::obtain_from_generic(m_method);
			feat=fitc_method->get_latent_features();
			SG_UNREF(fitc_method);
		}
		else
			feat=m_method->get_features();

		result=new CRegressionLabels(get_mean_vector(feat));

		SG_UNREF(feat);
	}
	else
	{
		result=new CRegressionLabels(get_mean_vector(data));
	}

	return result;
}

bool CGaussianProcessRegression::train_machine(CFeatures* data)
{
	// check whether given combination of inference method and likelihood function
	// supports regression
	REQUIRE(m_method, "Inference method must be attached\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_regression(), "%s with %s doesn't support regression\n",
			m_method->get_name(), lik->get_name())
	SG_UNREF(lik);

	if (data)
	{
		// set latent features for FITC inference method
		if (m_method->get_inference_type()==INF_FITC)
		{
			CFITCInferenceMethod* fitc_method=CFITCInferenceMethod::obtain_from_generic(m_method);
			fitc_method->set_latent_features(data);
			SG_UNREF(fitc_method);
		}
		else
			m_method->set_features(data);
	}

	// perform inference
	m_method->update_all();

	return true;
}

SGVector<float64_t> CGaussianProcessRegression::get_mean_vector(CFeatures* data)
{
	// check whether given combination of inference method and likelihood function
	// supports regression
	REQUIRE(m_method, "Inference method must be attached\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_regression(), "%s with %s doesn't support regression\n",
			m_method->get_name(), lik->get_name())
	SG_UNREF(lik);

	// check testing features
	REQUIRE(data, "Testing features can not be NULL\n")
	REQUIRE(data->has_property(FP_DOT),
			"Testing features must be type of CFeatures\n")
	REQUIRE(data->get_feature_class()==C_DENSE, "Testing features must be dense\n")
	REQUIRE(data->get_feature_type()==F_DREAL, "Testing features must be real\n")

	CFeatures* feat;

	// use latent features for FITC inference method
	if (m_method->get_inference_type()==INF_FITC)
	{
		CFITCInferenceMethod* fitc_method=CFITCInferenceMethod::obtain_from_generic(m_method);
		feat=fitc_method->get_latent_features();
		SG_UNREF(fitc_method);
	}
	else
		feat=m_method->get_features();

	// get kernel and compute kernel matrix: K(feat, data)*scale^2
	CKernel* kernel=m_method->get_kernel();
	kernel->init(feat, data);

	// get kernel matrix and create eigen representation of it
	SGMatrix<float64_t> k_trts=kernel->get_kernel_matrix();
	Map<MatrixXd> eigen_Ks(k_trts.matrix, k_trts.num_rows, k_trts.num_cols);

	// compute Ks=Ks*scale^2
	eigen_Ks*=CMath::sq(m_method->get_scale());

	// cleanup
	SG_UNREF(feat);
	SG_UNREF(kernel);

	if (data->get_feature_class()==C_COMBINED)
	{
		SG_WARNING("%s::get_mean_vector(): This only works for combined"
			" features which all share the same underlying object!\n",
			get_name())
		data=((CCombinedFeatures*)data)->get_first_feature_obj();
	}
	else
		SG_REF(data);

	// compute feature matrix
	SGMatrix<float64_t> feature_matrix=((CDotFeatures*)data)->get_computed_dot_feature_matrix();
	SG_UNREF(data);

	// get alpha and create eigen representation of it
	SGVector<float64_t> alpha=m_method->get_alpha();
	Map<VectorXd> eigen_alpha(alpha.vector, alpha.vlen);

	// get mean and create eigen representation of it
	CMeanFunction* mean_function=m_method->get_mean();
	SGVector<float64_t> m=mean_function->get_mean_vector(feature_matrix);
	Map<VectorXd> eigen_m(m.vector, m.vlen);
	SG_UNREF(mean_function);

	// compute mean: mu=Ks'*alpha+m
	SGVector<float64_t> mu(m.vlen);
	Map<VectorXd> eigen_mu(mu.vector, mu.vlen);
	eigen_mu=eigen_Ks.adjoint()*eigen_alpha+eigen_m;

	// evaluate mean
	lik=m_method->get_model();
	mu=lik->evaluate_means(mu);
	SG_UNREF(lik);

	return mu;
}

SGVector<float64_t> CGaussianProcessRegression::get_variance_vector(CFeatures* data)
{
	// check whether given combination of inference method and likelihood function
	// supports regression
	REQUIRE(m_method, "Inference method must be attached\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_regression(), "%s with %s doesn't support regression\n",
			m_method->get_name(), lik->get_name())

	// check testing features
	REQUIRE(data, "Testing features can not be NULL\n")
	REQUIRE(data->has_property(FP_DOT),
			"Testing features must be type of CFeatures\n")
	REQUIRE(data->get_feature_class()==C_DENSE, "Testing features must be dense\n")
	REQUIRE(data->get_feature_type()==F_DREAL, "Testing features must be real\n")

	CFeatures* feat;

	// use latent features for FITC inference method
	if (m_method->get_inference_type()==INF_FITC)
	{
		CFITCInferenceMethod* fitc_method=CFITCInferenceMethod::obtain_from_generic(m_method);
		feat=fitc_method->get_latent_features();
		SG_UNREF(fitc_method);
	}
	else
		feat=m_method->get_features();

	SG_REF(data);

	// get kernel and compute kernel matrix: K(data, data)*scale^2
	CKernel* kernel=m_method->get_kernel();
	kernel->init(data, data);

	// get kernel matrix and create eigen representation of it
	SGMatrix<float64_t> k_tsts=kernel->get_kernel_matrix();
	Map<MatrixXd> eigen_Kss(k_tsts.matrix, k_tsts.num_rows, k_tsts.num_cols);

	// compute Kss=Kss*scale^2
	eigen_Kss*=CMath::sq(m_method->get_scale());

	kernel->cleanup();

	// compute kernel matrix: K(feat, data)*scale^2
	kernel->init(feat, data);

	// get kernel matrix and create eigen representation of it
	SGMatrix<float64_t> k_trts=kernel->get_kernel_matrix();
	Map<MatrixXd> eigen_Ks(k_trts.matrix, k_trts.num_rows, k_trts.num_cols);

	// compute Ks=Ks*scale^2
	eigen_Ks*=CMath::sq(m_method->get_scale());

	// cleanup
	SG_UNREF(kernel);
	SG_UNREF(feat);
	SG_UNREF(data);

	// get shogun representation of cholesky and create eigen representation
	SGMatrix<float64_t> L=m_method->get_cholesky();
	Map<MatrixXd> eigen_L(L.matrix, L.num_rows, L.num_cols);

	// result variance vector
	SGVector<float64_t> s2(k_tsts.num_cols);
	Map<VectorXd> eigen_s2(s2.vector, s2.vlen);

	if (eigen_L.isUpperTriangular())
	{
		// get shogun of diagonal sigma vector and create eigen representation
		SGVector<float64_t> sW=m_method->get_diagonal_vector();
		Map<VectorXd> eigen_sW(sW.vector, sW.vlen);

		// solve L' * V = sW * Ks and compute V.^2
		MatrixXd eigen_V=eigen_L.triangularView<Upper>().adjoint().solve(
			eigen_sW.asDiagonal()*eigen_Ks);
		MatrixXd eigen_sV=eigen_V.cwiseProduct(eigen_V);

		eigen_s2=eigen_Kss.diagonal()-eigen_sV.colwise().sum().adjoint();
	}
	else
	{
		// M = Ks .* (L * Ks)
		MatrixXd eigen_M=eigen_Ks.cwiseProduct(eigen_L*eigen_Ks);
		eigen_s2=eigen_Kss.diagonal()+eigen_M.colwise().sum().adjoint();
	}

	// evaluate variance
	s2=lik->evaluate_variances(s2);
	SG_UNREF(lik);

	return s2;
}

bool CGaussianProcessRegression::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool CGaussianProcessRegression::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

#endif
