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

/** @brief The Gaussian Exact Form Inference Method.
 *
 *  This inference method computes the Gaussian Method
 *  exactly using matrix equations.
 *  \f[
 *  	 L = cholesky(K + \sigma^{2}I)
 *  \f]
 *
 *	L is the cholesky decomposition of K, the covariance matrix, plus
 *	a diagonal matrix with entries \f$\sigma\f$, the observation noise.
 *
 *  \f[
 *  	\boldsymbol{\alpha} = L^{T} \backslash(L \backslash \boldsymbol{y}})
 *  \f]
 *
 *  Where \f$L\f$ is the matrix mentioned above, \f$\boldsymbol{y}\f$ are the labels, and
 *  \f$\backslash\f$ is an operator (\f$x = A \backslash B\f$ means \f$Ax=B\f$.)
 *
 *  The Gaussian Likelihood Function must be used for this inference method.
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
	virtual EInferenceType get_inference_type() { return INF_EXACT; }

	/** returns the name of the inference method
	 *
	 * @return name Exact
	 */
	virtual const char* get_name() const { return "ExactInferenceMethod"; }

	/** get negative log marginal likelihood
	 *
	 * @return the negative log of the marginal likelihood function:
	 *
	 * \f[
	 *	  -log(p(y|X, \theta))
	 * \f]
	 *
	 * where \f$y\f$ are the labels, \f$X\f$ are the features,
	 * and \f$\theta\f$ represent hyperparameters.
	 */
	virtual float64_t get_negative_marginal_likelihood();

	/** get log marginal likelihood gradient
	 *
	 * @return vector of the  marginal likelihood function gradient
	 * with respect to hyperparameters:
	 *
	 * \f[
	 *	 -\frac{\partial {log(p(y|X, \theta))}}{\partial \theta}
	 * \f]
	 *
	 * where \f$y\f$ are the labels, \f$X\f$ are the features,
	 * and \f$\theta\f$ represent hyperparameters.
	 */
	virtual CMap<TParameter*, SGVector<float64_t> > get_marginal_likelihood_derivatives(
			CMap<TParameter*, CSGObject*>& para_dict);

	/** get alpha vector
	 *
	 * @return vector to compute posterior mean of Gaussian Process:
	 *
	 * \f[
	 *		\mu = K\alpha
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
	 *		 L = Cholesky(sW*K*sW+I)
	 * \f]
	 *
	 * where \f$K\f$ is the prior covariance matrix, \f$sW\f$ is the vector returned by
	 * get_diagonal_vector(), and \f$I\f$ is the identity matrix.
	 */
	virtual SGMatrix<float64_t> get_cholesky();

	/** get diagonal vector
	 *
	 * @return diagonal of matrix used to calculate posterior covariance matrix
	 *
	 * \f[
	 *	    Cov = (K^{-1}+sW^{2})^{-1}
	 * \f]
	 *
	 * where \f$Cov\f$ is the posterior covariance matrix, \f$K\f$ is
	 * the prior covariance matrix, and \f$sW\f$ is the diagonal vector.
	 */
	virtual SGVector<float64_t> get_diagonal_vector();

	/** get the gradient
	 *
	 * @return map of gradient: keys are names of parameters, values are
	 * values of derivative with respect to that parameter.
	 */
	virtual CMap<TParameter*, SGVector<float64_t> > get_gradient(
			CMap<TParameter*, CSGObject*>& para_dict)
	{
		return get_marginal_likelihood_derivatives(para_dict);
	}

	/** get the function value
	 *
	 * @return vector that represents the function value
	 */
	virtual SGVector<float64_t> get_quantity()
	{
		SGVector<float64_t> result(1);
		result[0] = get_negative_marginal_likelihood();
		return result;
	}

	/**
	 * @return wether combination of exact inference method and given likelihood
	 * function supports regression
	 */
	virtual bool supports_regression()
	{
		check_members();
		return m_model->supports_regression();
	}

	/** update all matrices */
	virtual void update_all();

protected:
	/** update alpha matrix */
	virtual void update_alpha();

	/** update Cholesky matrix */
	virtual void update_chol();

	/** update kernel matrix */
	virtual void update_train_kernel();

private:
	/** Check if members of object are valid
	 * for inference
	 */
	void check_members();
};
}
#endif // HAVE_EIGEN3
#endif /* CEXACTINFERENCEMETHOD_H_ */
