/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Wu Lin
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 * Copyright (C) 2015 Wu Lin
 *
 * Code adapted from Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */

#ifndef CFITCINFERENCEMETHOD_H
#define CFITCINFERENCEMETHOD_H

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/SingleFITCLaplacianBase.h>

namespace shogun
{

/** @brief The Fully Independent Conditional Training inference method class.
 *
 * This inference method computes the Cholesky and Alpha vectors approximately
 * with the help of inducing variables. For more details, see "Sparse Gaussian
 * Process using Pseudo-inputs", Edward Snelson, Zoubin Ghahramani, NIPS 18, MIT
 * Press, 2005.
 *
 * This specific implementation was inspired by the infFITC.m file in the GPML
 * toolbox.
 *
 * NOTE: The Gaussian Likelihood Function must be used for this inference
 * method.
 *
 * Note that the number of inducing points (m) is usually far less than the number of input points (n).
 * (the time complexity is computed based on the assumption m < n)
 *
 * Warning: the time complexity of method,
 * CSingleFITCLaplacianBase::get_derivative_wrt_kernel(const TParameter* param),
 * depends on the implementation of kernel method, get_parameter_gradient_diagonal(param, i).
 * The default time complexity of the kernel method can be O(n^2)
 *
 * Warning: the the time complexity increases from O(m^2*n) to O(n^2*m) if method 
 * CFITCInferenceMethod::get_posterior_covariance() is called
 */
class CFITCInferenceMethod: public CSingleFITCLaplacianBase
{
public:
	/** default constructor */
	CFITCInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model likelihood model to use
	 * @param inducing_features features to use
	 */
	CFITCInferenceMethod(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model,
			CFeatures* inducing_features);

	virtual ~CFITCInferenceMethod();

	/** returns the name of the inference method
	 *
	 * @return name FITC
	 */
	virtual const char* get_name() const { return "FITCInferenceMethod"; }

	/** return what type of inference we are
	 *
	 * @return inference type FITC_REGRESSION
	 */
	virtual EInferenceType get_inference_type() const { return INF_FITC_REGRESSION; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted CFITCInferenceMethod object
	 */
	static CFITCInferenceMethod* obtain_from_generic(CInferenceMethod* inference);

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

	/**
	 * @return whether combination of FITC inference method and given likelihood
	 * function supports regression
	 */
	virtual bool supports_regression() const
	{
		check_members();
		return m_model->supports_regression();
	}


	/** returns mean vector \f$\mu\f$ of the Gaussian distribution
	 * \f$\mathcal{N}(\mu,\Sigma)\f$, which is an approximation to the
	 * posterior:
	 *
	 * \f[
	 * p(f|y) \approx q(f|y) = \mathcal{N}(\mu,\Sigma)
	 * \f]
	 *
	 * in case if particular inference method doesn't compute posterior
	 * \f$p(f|y)\f$ exactly, and it returns covariance matrix \f$\Sigma\f$ of
	 * the posterior Gaussian distribution \f$\mathcal{N}(\mu,\Sigma)\f$
	 * otherwise.
	 *
	 * @return mean vector
	 */
	virtual SGVector<float64_t> get_posterior_mean();

	/** returns covariance matrix \f$\Sigma\f$ of the Gaussian distribution
	 * \f$\mathcal{N}(\mu,\Sigma)\f$, which is an approximation to the
	 * posterior:
	 *
	 * \f[
	 * p(f|y) \approx q(f|y) = \mathcal{N}(\mu,\Sigma)
	 * \f]
	 *
	 * in case if particular inference method doesn't compute posterior
	 * \f$p(f|y)\f$ exactly, and it returns covariance matrix \f$\Sigma\f$ of
	 * the posterior Gaussian distribution \f$\mathcal{N}(\mu,\Sigma)\f$
	 * otherwise.
	 *
	 * @return covariance matrix
	 */
	virtual SGMatrix<float64_t> get_posterior_covariance();

	/** update all matrices */
	virtual void update();
protected:
	/** check if members of object are valid for inference */
	virtual void check_members() const;

	/** update alpha matrix */
	virtual void update_alpha();

	/** update cholesky Matrix.*/
	virtual void update_chol();

	/** update matrices which are required to compute negative log marginal
	 * likelihood derivatives wrt hyperparameter
	 */
	virtual void update_deriv();

	/** returns derivative of negative log marginal likelihood wrt parameter of
	 * likelihood model
	 *
	 * @param param parameter of given likelihood model
	 *
	 * @return derivative of negative log marginal likelihood
	 */
	virtual SGVector<float64_t> get_derivative_wrt_likelihood_model(
			const TParameter* param);

protected:
	/** Cholesky of covariance of inducing features */
	SGMatrix<float64_t> m_chol_uu;

	/** Cholesky of covariance of inducing features and training features */
	SGMatrix<float64_t> m_chol_utr;

	/** labels adjusted for noise and means */
	SGVector<float64_t> m_r;

	/** solves the equation V * r = m_chol_utr */
	SGVector<float64_t> m_be;

private:
	void init();
};
}
#endif /* HAVE_EIGEN3 */
#endif /* CFITCINFERENCEMETHOD_H */
