/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CEXACTINFERENCEMETHOD_H_
#define CEXACTINFERENCEMETHOD_H_

#include <shogun/lib/config.h>
#include <shogun/regression/gp/InferenceMethod.h>

#ifdef HAVE_LAPACK
namespace shogun
{

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
 *
 *
 *  The Gaussian Likelihood Function must be used for this inference method.
 *
 */
class CExactInferenceMethod: public CInferenceMethod
{

public:

	/*Default Constructor*/
	CExactInferenceMethod();

	/* Constructor
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	CExactInferenceMethod(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

	/*Destructor*/
	virtual ~CExactInferenceMethod();

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
	virtual CMap<TParameter*, float64_t> get_marginal_likelihood_derivatives(
			CMap<TParameter*, CSGObject*>& para_dict);

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


	/** get Cholesky Decomposition Matrix
	 *
	 * @return Cholesky Decomposition of Matrix:
	 * \f[
	 *		 L = Cholesky(sW*K*sW+I)
	 * \f]
	 *
	 * 	Where K is the prior covariance matrix, sW is the matrix returned by
	 * 	get_cholesky(), and I is the identity matrix.
	 */
	virtual SGMatrix<float64_t> get_cholesky();

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

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 * @return name of the SGSerializable
	 */
	inline virtual const char* get_name() const
	{
		return "ExactInferenceMethod";
	}
	
	/*Get the gradient
	 *
	 * @return Map of gradient. Keys are names of parameters, values are
	 * values of derivative with respect to that parameter.
	 */
	virtual CMap<TParameter*, float64_t> get_gradient(
			CMap<TParameter*, CSGObject*>& para_dict)
	{
		return get_marginal_likelihood_derivatives(para_dict);
	}

	/*Get the function value
	 *
	 * @return Vector that represents the function value
	 */
	virtual SGVector<float64_t> get_quantity()
	{
		SGVector<float64_t> result(1);
		result[0] = get_negative_marginal_likelihood();
		return result;
	}

protected:
	/** Update Alpha and Cholesky Matrices.
	 */
	virtual void update_alpha();
	virtual void update_chol();
	virtual void update_train_kernel();
	virtual void update_all();

private:

	/** Check if members of object are valid
	 * for inference
	 */
	void check_members();

	/*Kernel matrix with noise*/
	SGMatrix<float64_t> m_kern_with_noise;
};

}
#endif // HAVE_LAPACK

#endif /* CEXACTINFERENCEMETHOD_H_ */
