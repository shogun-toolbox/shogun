/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 */

#ifndef CEXACTINFERENCEMETHOD_H_
#define CEXACTINFERENCEMETHOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/InferenceMethod.h>

namespace shogun
{

/** @brief The Gaussian exact form inference method class.
 *
 * This inference method computes the Gaussian Method exactly using matrix
 * equations.
 *
 * \f[
 * L = cholesky(K + \sigma^{2}I)
 * \f]
 *
 * \f$L\f$ is the cholesky decomposition of \f$K\f$, the covariance matrix, plus
 * a diagonal matrix with entries \f$\sigma^{2}\f$, the observation noise.
 *
 * \f[
 * \boldsymbol{\alpha} = L^{T} \backslash(L \backslash \boldsymbol{y}})
 * \f]
 *
 * where \f$L\f$ is the matrix mentioned above, \f$\boldsymbol{y}\f$ are the
 * labels, and \f$\backslash\f$ is an operator (\f$x = A \backslash B\f$ means
 * \f$Ax=B\f$.)
 *
 * NOTE: The Gaussian Likelihood Function must be used for this inference
 * method.
 */
class CExactInferenceMethod: public CInferenceMethod
{
public:
	/** default constructor */
	CExactInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function to use
	 * @param labels labels of the features
	 * @param model likelihood model to use
	 */
	CExactInferenceMethod(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

	virtual ~CExactInferenceMethod();

	/** return what type of inference we are
	 *
	 * @return inference type EXACT
	 */
	virtual EInferenceType get_inference_type() const { return INF_EXACT; }

	/** returns the name of the inference method
	 *
	 * @return name Exact
	 */
	virtual const char* get_name() const { return "ExactInferenceMethod"; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted CExactInferenceMethod object
	 */
	static CExactInferenceMethod* obtain_from_generic(CInferenceMethod* inference);

	/** get negative log marginal likelihood
	 *
	 * @return the negative log of the marginal likelihood function:
	 *
	 * \f[
	 * -log(p(y|X, \theta))
	 * \f]
	 *
	 * where \f$y\f$ are the labels, \f$X\f$ are the features, and \f$\theta\f$
	 * represent hyperparameters.
	 */
	virtual float64_t get_negative_log_marginal_likelihood();

	/** get alpha vector
	 *
	 * @return vector to compute posterior mean of Gaussian Process:
	 *
	 * \f[
	 * \mu = K\alpha
	 * \f]
	 *
	 * where \f$\mu\f$ is the mean and \f$K\f$ is the prior covariance matrix.
	 */
	virtual SGVector<float64_t> get_alpha();

	/** get Cholesky decomposition matrix
	 *
	 * @return Cholesky decomposition of matrix:
	 *
	 * \f[
	 * L = Cholesky(sW*K*sW+I)
	 * \f]
	 *
	 * where \f$K\f$ is the prior covariance matrix, \f$sW\f$ is the vector
	 * returned by get_diagonal_vector(), and \f$I\f$ is the identity matrix.
	 */
	virtual SGMatrix<float64_t> get_cholesky();

	/** get diagonal vector
	 *
	 * @return diagonal of matrix used to calculate posterior covariance matrix
	 *
	 * \f[
	 * Cov = (K^{-1}+sW^{2})^{-1}
	 * \f]
	 *
	 * where \f$Cov\f$ is the posterior covariance matrix, \f$K\f$ is the prior
	 * covariance matrix, and \f$sW\f$ is the diagonal vector.
	 */
	virtual SGVector<float64_t> get_diagonal_vector();

	/** returns mean vector \f$\mu\f$ of the posterior Gaussian distribution
	 * \f$\mathcal{N}(\mu,\Sigma)\f$
	 *
	 * \f[
	 * p(f|y) = \mathcal{N}(\mu,\Sigma)
	 * \f]
	 *
	 * @return mean vector
	 */
	virtual SGVector<float64_t> get_posterior_mean();

	/** returns covariance matrix \f$\Sigma\f$ of the posterior Gaussian
	 * distribution \f$\mathcal{N}(\mu,\Sigma)\f$
	 *
	 * \f[
	 * p(f|y) = \mathcal{N}(\mu,\Sigma)
	 * \f]
	 *
	 * @return covariance matrix
	 */
	virtual SGMatrix<float64_t> get_posterior_covariance();

	/**
	 * @return whether combination of exact inference method and given
	 * likelihood function supports regression
	 */
	virtual bool supports_regression() const
	{
		check_members();
		return m_model->supports_regression();
	}

	/** update all matrices */
	virtual void update();

protected:
	/** check if members of object are valid for inference */
	virtual void check_members() const;

	/** update alpha matrix */
	virtual void update_alpha();

	/** update Cholesky matrix */
	virtual void update_chol();

	/** update mean vector of the posterior Gaussian */
	virtual void update_mean();

	/** update covariance matrix of the posterior Gaussian */
	virtual void update_cov();

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
	/** covariance matrix of the the posterior Gaussian distribution */
	SGMatrix<float64_t> m_Sigma;

	/** mean vector of the the posterior Gaussian distribution */
	SGVector<float64_t> m_mu;

	SGMatrix<float64_t> m_Q;
};
}
#endif /* HAVE_EIGEN3 */
#endif /* CEXACTINFERENCEMETHOD_H_ */
