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

#ifndef CSINGLELAPLACIANINFERENCEMETHOD_H_
#define CSINGLELAPLACIANINFERENCEMETHOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/LaplacianInferenceBase.h>

namespace shogun
{

/** @brief The SingleLaplace approximation inference method class
 * for regression and binary Classification.
 *
 * This inference method approximates the posterior likelihood function by using
 * Laplace's method. Here, we compute a Gaussian approximation to the posterior
 * via a Taylor expansion around the maximum of the posterior likelihood
 * function.
 *
 * For more details, see "Bayesian Classification with Gaussian Processes" by
 * Christopher K.I Williams and David Barber, published 1998 in the IEEE
 * Transactions on Pattern Analysis and Machine Intelligence, Volume 20, Number
 * 12, Pages 1342-1351.
 *
 * This specific implementation was adapted from the infLaplace.m file in the
 * GPML toolbox.
 */
class CSingleLaplacianInferenceMethod: public CLaplacianInferenceBase
{
public:
	/** default constructor */
	CSingleLaplacianInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	CSingleLaplacianInferenceMethod(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

	virtual ~CSingleLaplacianInferenceMethod();

	/** returns the name of the inference method
	 *
	 * @return name SingleLaplacian
	 */
	virtual const char* get_name() const { return "SingleLaplacianInferenceMethod"; }

	/** return what type of inference we are
	 *
	 * @return inference type Laplacian_Single
	 */
	virtual EInferenceType get_inference_type() const { return INF_LAPLACIAN_SINGLE; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted CSingleLaplacianInferenceMethod object
	 */
	static CSingleLaplacianInferenceMethod* obtain_from_generic(CInferenceMethod* inference);

	/** get negative log marginal likelihood
	 *
	 * @return the negative log of the marginal likelihood function:
	 *
	 * \f[
	 * -log(p(y|X, \theta))
	 * \f]
	 *
	 * where \f$y\f$ are the labels, \f$X\f$ are the features, and
	 * \f$\theta\f$ represent hyperparameters.
	 */
	virtual float64_t get_negative_log_marginal_likelihood();

	/**
	 * @return whether combination of Laplace approximation inference method and
	 * given likelihood function supports regression
	 */
	virtual bool supports_regression() const
	{
		check_members();
		return m_model->supports_regression();
	}

	/**
	 * @return whether combination of Laplace approximation inference method and
	 * given likelihood function supports binary classification
	 */
	virtual bool supports_binary() const
	{
		check_members();
		return m_model->supports_binary();
	}

	/** get diagonal vector
	 *
	 * @return diagonal of matrix used to calculate posterior covariance matrix:
	 *
	 * \f[
	 * Cov = (K^{-1}+sW^{2})^{-1}
	 * \f]
	 *
	 * where \f$Cov\f$ is the posterior covariance matrix, \f$K\f$ is the prior
	 * covariance matrix, and \f$sW\f$ is the diagonal vector.
	 */
	virtual SGVector<float64_t> get_diagonal_vector();

	/** update data all matrices */
	virtual void update();
protected:

	virtual void update_init();

	/** update alpha matrix */
	virtual void update_alpha();

	/** update cholesky matrix */
	virtual void update_chol();

	/** update covariance matrix of the approximation to the posterior */
	virtual void update_approx_cov();

	/** update matrices which are required to compute negative log marginal
	 * likelihood derivatives wrt hyperparameter
	 */
	virtual void update_deriv();

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * CInferenceMethod class
	 *
	 * @param param parameter of CInferenceMethod class
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_inference_method(
			const TParameter* param);

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * likelihood model
	 *
	 * @param param parameter of given likelihood model
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_likelihood_model(
			const TParameter* param);

	/** returns derivative of negative log marginal likelihood wrt kernel's
	 * parameter
	 *
	 * @param param parameter of given kernel
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_kernel(
			const TParameter* param);

	/** returns derivative of negative log marginal likelihood wrt mean
	 * function's parameter
	 *
	 * @param param parameter of given mean function
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_mean(
			const TParameter* param);

private:
	void init();

protected:
	/** square root of W */
	SGVector<float64_t> m_sW;

	/** second derivative of log likelihood with respect to function location */
	SGVector<float64_t> m_d2lp;

	/** third derivative of log likelihood with respect to function location */
	SGVector<float64_t> m_d3lp;

	SGVector<float64_t> m_dfhat;

	SGMatrix<float64_t> m_Z;

	SGVector<float64_t> m_g;

	float64_t m_Psi;
};
}
#endif /* HAVE_EIGEN3 */
#endif /* CSINGLELAPLACIANINFERENCEMETHOD_H_ */
