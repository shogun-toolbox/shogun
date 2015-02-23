/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 *
 * Based on ideas from GAUSSIAN PROCESS REGRESSION AND CLASSIFICATION Toolbox
 * Copyright (C) 2005-2013 by Carl Edward Rasmussen & Hannes Nickisch under the
 * FreeBSD License
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */

#ifndef _EPINFERENCEMETHOD_H_
#define _EPINFERENCEMETHOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/InferenceMethod.h>

namespace shogun
{

/** @brief Class of the Expectation Propagation (EP) posterior approximation
 * inference method.
 *
 * For more details, see: Minka, T. P. (2001). A Family of Algorithms for
 * Approximate Bayesian Inference. PhD thesis, Massachusetts Institute of
 * Technology
 */
class CEPInferenceMethod : public CInferenceMethod
{
public:
	/** default constructor */
	CEPInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model likelihood model to use
	 */
	CEPInferenceMethod(CKernel* kernel, CFeatures* features, CMeanFunction* mean,
			CLabels* labels, CLikelihoodModel* model);

	virtual ~CEPInferenceMethod();

	/** return what type of inference we are
	 *
	 * @return inference type EP
	 */
	virtual EInferenceType get_inference_type() const { return INF_EP; }

	/** returns the name of the inference method
	 *
	 * @return name EP
	 */
	virtual const char* get_name() const { return "EPInferenceMethod"; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted CEPInferenceMethod object
	 */
	static CEPInferenceMethod* obtain_from_generic(CInferenceMethod* inference);

	/** returns the negative logarithm of the marginal likelihood function:
	 *
	 * \f[
	 * -log(p(y|X, \theta))
	 * \f]
	 *
	 * where \f$y\f$ are the labels, \f$X\f$ are the features, and \f$\theta\f$
	 * represent hyperparameters.
	 *
	 * @return negative log marginal likelihood
	 */
	virtual float64_t get_negative_log_marginal_likelihood();

	/** returns vector to compute posterior mean of Gaussian Process under EP
	 * approximation:
	 *
	 * \f[
	 * \mathbb{E}_q[f_*|X,y,x_*] = k^T_*\alpha
	 * \f]
	 *
	 * where \f$k^T_*\f$ - covariance between training points \f$X\f$ and test
	 * point \f$x_*\f$, and for EP approximation:
	 *
	 * \f[
	 * \alpha = (K + \tilde{S}^{-1})^{-1}\tilde{S}^{-1}\tilde{\nu} =
	 * (I-\tilde{S}^{\frac{1}{2}}B^{-1}\tilde{S}^{\frac{1}{2}}K)\tilde{\nu}
	 * \f]
	 *
	 * where \f$K\f$ is the prior covariance matrix,
	 * \f$\tilde{S}^{\frac{1}{2}}\f$ is the diagonal matrix (see description of
	 * get_diagonal_vector() method) and \f$\tilde{\nu}\f$ - natural parameter
	 * (\f$\tilde{\nu} = \tilde{S}\tilde{\mu}\f$).
	 *
	 * @return vector \f$\alpha\f$
	 */
	virtual SGVector<float64_t> get_alpha();

	/** returns upper triangular factor \f$L^T\f$ of the Cholesky decomposition
	 * (\f$LL^T\f$) of the matrix:
	 *
	 * \f[
	 * B = (\tilde{S}^{\frac{1}{2}}K\tilde{S}^{\frac{1}{2}}+I)
	 * \f]
	 *
	 * where \f$\tilde{S}^{\frac{1}{2}}\f$ is the diagonal matrix (see
	 * description of get_diagonal_vector() method) and \f$K\f$ is the prior
	 * covariance matrix.

	 * @return upper triangular factor of the Cholesky decomposition of the
	 * matrix \f$B\f$
	 */
	virtual SGMatrix<float64_t> get_cholesky();

	/** returns diagonal vector of the diagonal matrix:
	 *
	 * \f[
	 * \tilde{S}^{\frac{1}{2}} = \sqrt{\tilde{S}}
	 * \f]
	 *
	 * where \f$\tilde{S} = \text{diag}(\tilde{\tau})\f$, and \f$\tilde{\tau}\f$
	 * - natural parameter (\f$\tilde{\tau}_i = \tilde{\sigma}_i^{-2}\f$).
	 *
	 * @return diagonal vector of the matrix \f$\tilde{S}^{\frac{1}{2}}\f$
	 */
	virtual SGVector<float64_t> get_diagonal_vector();

	/** returns mean vector \f$\mu\f$ of the Gaussian distribution
	 * \f$\mathcal{N}(\mu,\Sigma)\f$, which is an approximation to the
	 * posterior:
	 *
	 * \f[
	 * p(f|X,y) \approx q(f|X,y) = \mathcal{N}(f|\mu,\Sigma)
	 * \f]
	 *
	 * Mean vector \f$\mu\f$ is evaluated like:
	 *
	 * \f[
	 * \mu = \Sigma\tilde{\nu}
	 * \f]
	 *
	 * where \f$\Sigma\f$ - covariance matrix of the posterior approximation and
	 * \f$\tilde{\nu}\f$ - natural parameter (\f$\tilde{\nu} =
	 * \tilde{S}\tilde{\mu}\f$).
	 *
	 * @return mean vector \f$\mu\f$
	 */
	virtual SGVector<float64_t> get_posterior_mean();

	/** returns covariance matrix \f$\Sigma=(K^{-1}+\tilde{S})^{-1}\f$ of the
	 * Gaussian distribution \f$\mathcal{N}(\mu,\Sigma)\f$, which is an
	 * approximation to the posterior:
	 *
	 * \f[
	 * p(f|X,y) \approx q(f|X,y) = \mathcal{N}(f|\mu,\Sigma)
	 * \f]
	 *
	 * Covariance matrix \f$\Sigma\f$ is evaluated using matrix inversion lemma:
	 *
	 * \f[
	 * \Sigma = (K^{-1}+\tilde{S})^{-1} = K -
	 * K\tilde{S}^{\frac{1}{2}}B^{-1}\tilde{S}^{\frac{1}{2}}K
	 * \f]
	 *
	 * where \f$B=(\tilde{S}^{\frac{1}{2}}K\tilde{S}^{\frac{1}{2}}+I)\f$.
	 *
	 * @return covariance matrix \f$\Sigma\f$
	 */
	virtual SGMatrix<float64_t> get_posterior_covariance();

	/** returns tolerance of the EP approximation
	 *
	 * @return tolerance
	 */
	virtual float64_t get_tolerance() const { return m_tol; }

	/** sets tolerance of the EP approximation
	 *
	 * @param tol tolerance to set
	 */
	virtual void set_tolerance(const float64_t tol) { m_tol=tol; }

	/** returns minimum number of sweeps over all variables
	 *
	 * @return minimum number of sweeps
	 */
	virtual uint32_t get_min_sweep() const { return m_min_sweep; }

	/** sets minimum number of sweeps over all variables
	 *
	 * @param min_sweep minimum number of sweeps to set
	 */
	virtual void set_min_sweep(const uint32_t min_sweep) { m_min_sweep=min_sweep; }

	/** returns maximum number of sweeps over all variables
	 *
	 * @return maximum number of sweeps
	 */
	virtual uint32_t get_max_sweep() const { return m_max_sweep; }

	/** sets maximum number of sweeps over all variables
	 *
	 * @param max_sweep maximum number of sweeps to set
	 */
	virtual void set_max_sweep(const uint32_t max_sweep) { m_max_sweep=max_sweep; }

	/**
	 * @return whether combination of Laplace approximation inference method and
	 * given likelihood function supports binary classification
	 */
	virtual bool supports_binary() const
	{
		check_members();
		return m_model->supports_binary();
	}

	/** update data all matrices */
	virtual void update();

protected:
	/** update alpha matrix */
	virtual void update_alpha();

	/** update Cholesky matrix */
	virtual void update_chol();

	/** update covariance matrix of the approximation to the posterior */
	virtual void update_approx_cov();

	/** update mean vector of the approximation to the posterior */
	virtual void update_approx_mean();

	/** update negative marginal likelihood */
	virtual void update_negative_ml();

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

private:
	/** mean vector of the approximation to the posterior */
	SGVector<float64_t> m_mu;

	/** covariance matrix of the approximation to the posterior */
	SGMatrix<float64_t> m_Sigma;

	/** negative marginal likelihood */
	float64_t m_nlZ;

	/** vector of natural parameters \f$\tilde{\nu} = \tilde{S}\tilde{\mu}\f$,
	 * where \f$\tilde{S} = \text{diag}(\tilde{\tau})\f$
	 */
	SGVector<float64_t> m_tnu;

	/** vector of natural parameters \f$\tilde{\tau}_i =
	 * \tilde{\sigma}_i^{-2}\f$
	 */
	SGVector<float64_t> m_ttau;

	/** square root of the \f$\tilde{\tau}\f$ vector */
	SGVector<float64_t> m_sttau;

	/** tolerance of the EP approximation */
	float64_t m_tol;

	/** minimum number of sweeps over all variables */
	uint32_t m_min_sweep;

	/** maximum number of sweeps over all variables */
	uint32_t m_max_sweep;

	SGMatrix<float64_t> m_F;
};
}
#endif /* HAVE_EIGEN3 */
#endif /* _EPINFERENCEMETHOD_H_ */
