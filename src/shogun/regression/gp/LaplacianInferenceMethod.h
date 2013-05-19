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

#ifndef CLAPLACIANINFERENCEMETHOD_H_
#define CLAPLACIANINFERENCEMETHOD_H_

#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3

#include <shogun/regression/gp/InferenceMethod.h>

namespace shogun
{

/** @brief The Laplace Approximation
 *  Inference Method
 *
 *  This inference method approximates the
 *  posterior likelihood function by using
 *  Laplace's method. Here, we compute a Gaussian
 *  approximation to the posterior via a
 *  Taylor expansion around the maximum of the posterior
 *  likelihood function. For more details, see "Bayesian
 *  Classification with Gaussian Processes" by Christopher K.I
 *  Williams and David Barber, published 1998 in the IEEE
 *  Transactions on Pattern Analysis and Machine Intelligence,
 *  Volume 20, Number 12, Pages 1342-1351.
 *
 *
 *
 *  This specific implementation was adapted from the infLaplace.m file
 *  in the GPML toolbox
 *
 *
 */
class CLaplacianInferenceMethod: public CInferenceMethod
{

public:

	/** Default Constructor*/
	CLaplacianInferenceMethod();

	/** Constructor
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	CLaplacianInferenceMethod(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

	/** Destructor*/
	virtual ~CLaplacianInferenceMethod();

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
	virtual CMap<TParameter*, SGVector<float64_t> >
		get_marginal_likelihood_derivatives(CMap<TParameter*, CSGObject*>& para_dict);

	/** get Alpha Matrix
	 *
	 * @return Matrix to compute posterior mean of Gaussian Process:
	 * \f[
	 *		\mu = K\alpha
	 * \f]
	 *
	 * 	where \f$\mu\f$ is the mean and K is the prior covariance matrix
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
	virtual const char* get_name() const
	{
		return "LaplacianInferenceMethod";
	}

	/** Get the gradient
	 *
	 * @return Map of gradient. Keys are names of parameters, values are
	 * values of derivative with respect to that parameter.
	 */
	virtual CMap<TParameter*, SGVector<float64_t> > get_gradient(
			CMap<TParameter*, CSGObject*>& para_dict)
	{
		return get_marginal_likelihood_derivatives(para_dict);
	}

	/** Get the function value
	 *
	 * @return Vector that represents the function value
	 */
	virtual SGVector<float64_t> get_quantity()
	{
		SGVector<float64_t> result(1);
		result[0] = get_negative_marginal_likelihood();
		return result;
	}

	/** Get tolerance for Newton Iterations
	 *
	 * @return Tolerance for Newton Iterations
	 */
	virtual float64_t get_newton_tolerance() { return m_tolerance; }

	/** Set tolerance for Newton Iterations
	 *
	 * @param tol Tolerance for Newton Iterations
	 */
	virtual void set_newton_tolerance(float64_t tol) { m_tolerance=tol; }

	/** Get tolerance for Brent's Minimization Method
	 *
	 * @return tolerance for Brent's Minimization Method
	 */
	virtual float64_t get_minimization_tolerance() { return m_opt_tolerance; }

	/** Set tolerance for Brent's Minimization Method
	 *
	 * @param tol tolerance for Brent's Minimization Method
	 */
	virtual void set_minimization_tolerance(float64_t tol) { m_opt_tolerance=tol; }

	/** Get max iterations for Brent's Minimization Method
	 *
	 * @return max iterations for Brent's Minimization Method
	 */
	virtual int32_t get_minimization_iterations() { return m_max; }

	/** Set max iterations for Brent's Minimization Method
	 *
	 * @param itr max iterations for Brent's Minimization Method
	 */
	virtual void set_minimization_tolerance(int32_t itr) { m_max=itr; }

	/** Get max Newton iterations
	 *
	 * @return max Newton iterations
	 */
	virtual int32_t get_newton_iterations() { return m_max_itr; }

	/** Set max Newton iterations
	 *
	 * @param itr max Newton iterations
	 */
	virtual void set_newton_tolerance(int32_t itr) { m_max_itr=itr; }

protected:
	/** Update alpha matrix */
	virtual void update_alpha();

	/** Update cholesky Matrix.*/
	virtual void update_chol();

	/** Update train kernel matrix */
	virtual void update_train_kernel();

	/** Update data means */
	virtual void update_all();

private:

	void init();

private:

	/** Check if members of object are valid
	 * for inference
	 */
	void check_members();

	/*Amount of tolerance for Newton's Iterations*/
	float64_t m_tolerance;

	/*Amount of tolerance for Brent's Minimization Method*/
	float64_t m_opt_tolerance;

	/*Max iterations for Brent's Minimization Method*/
	float64_t m_max;

	/*Max Newton Iterations*/
	index_t m_max_itr;

	/*Eigen version of alpha vector*/
	SGVector<float64_t> temp_alpha;

	/*Function Location*/
	SGVector<float64_t> function;

	/*Noise Matrix*/
	SGVector<float64_t> W;

	/*Square root of W*/
	SGVector<float64_t> sW;

	/*Eigen version of means vector*/
	SGVector<float64_t> m_means;

	/*Derivative of log likelihood with respect
	 * to function location
	 */
	SGVector<float64_t> dlp;

	/*Second derivative of log likelihood with respect
	 * to function location
	 */
	SGVector<float64_t> d2lp;

	/*Third derivative of log likelihood with respect
	 * to function location
	 */
	SGVector<float64_t> d3lp;

	/*log likelihood*/
	float64_t lp;
};

}
#endif // HAVE_EIGEN3

#endif /* CLAPLACIANINFERENCEMETHOD_H_ */
