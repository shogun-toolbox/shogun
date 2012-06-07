/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef EXACTINFERENCEMETHOD_H_
#define EXACTINFERENCEMETHOD_H_

#include <shogun/regression/gp/InferenceMethod.h>

namespace shogun {

/** @brief The Gaussian Exact Form Inference Method
 *
 *  This inference method computes the Gaussian Method
 *  exactly using matrix equations.
 *  /f[
 *  	 L = cholesky(K + \sigma^{2}I)
 *  /f]
 *
 *	L is the cholesky decomposition of K, the covariance matrix, plus
 *	a diagonal matrix with entries $\sigma$, the observation noise.
 *
 *  /f[
 *  	\boldsymbol{\alpha} = L^{T} \backslash(L \backslash \boldsymbol{y}})
 *  /f]
 *
 *  Where L is the matrix mentioned above, $\boldsymbol{y}$ are the labels, and
 *  $\backslash$ is an operator ($x = A \backslash B$ means Ax=B.)
 *
 *  \f[
 *  \f]
 *
 *
 *  function must be used for this inference method.
 *
 */
class CExactInferenceMethod: public CInferenceMethod {
public:
	/*Default Constructor*/
	CExactInferenceMethod();

	~CExactInferenceMethod();

	/* Constructor
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	CExactInferenceMethod(CKernel* kernel, CDotFeatures* features,
			CLabels* labels, CLikelihoodModel* model);

	/*Temporary Function for learning parameters
	 * this will eventually be incorporated into
	 * the model selection framework.
	 */
	void learn_parameters();

	/** get Negative Log Marginal Likelihood
	 *
	 * @return The Negative Log of the Marginal Likelihood function:
	 * \f[
	 *	  -log(p(y|X, \theta))
	 *	  Where y are the labels, X are the features,
	 *	  and \theta represent hyperparameters
	 * \f]
	 */
	virtual float64_t get_negative_marginal_likelihood();

	/** get Log Marginal Likelihood Gradient
	 *
	 * @return Vector of the  Marginal Likelihood Function Gradient
	 *         with respect to hyperparameters
	 * \f[
	 *	 -\frac{\partial {log(p(y|X, \theta))}}{\partial \theta}
	 * \f]
	 */
	virtual SGVector<float64_t> get_marginal_likelihood_derivatives();

	/** get Alpha Matrix
	 *
	 * @return Matrix to compute posterior mean of Gaussian Process:
	 * \f[
	 *		\mu = K\alpha
	 * \f]
	 *
	 * 	where \mu is the mean and K is the prior covariance matrix
	 */
	virtual SGVector<float64_t> get_alpha();

	/** get Diagonal Vector
	 *
	 * @return Diagonal of matrix used to calculate posterior covariance matrix
	 * \f[
	 *	    Cov = (K^{-1}+D^{2})^{-1}}
	 * \f]
	 *
	 *  Where Cov is the posterior covariance matrix, K is
	 *  the prior covariance matrix, and D is the diagonal matrix
	 */
	virtual SGVector<float64_t> get_diagonal_vector();

	inline virtual const char* get_name() const { return "ExactInferenceMethod"; }

private:

	/** Check if members of object are valid
	 * for inference
	 */
	void check_members();
};

}

#endif /* EXACTINFERENCEMETHOD_H_ */
