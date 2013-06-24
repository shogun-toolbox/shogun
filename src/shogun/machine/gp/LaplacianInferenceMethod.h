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

#ifndef CLAPLACIANINFERENCEMETHOD_H_
#define CLAPLACIANINFERENCEMETHOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/InferenceMethod.h>

namespace shogun
{

/** @brief The Laplace Approximation Inference Method.
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
 *  This specific implementation was adapted from the infLaplace.m file
 *  in the GPML toolbox
 */
class CLaplacianInferenceMethod: public CInferenceMethod
{
public:
	/** default constructor */
	CLaplacianInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	CLaplacianInferenceMethod(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

	virtual ~CLaplacianInferenceMethod();

	/** return what type of inference we are
	 *
	 * @return inference type LAPLACIAN
	 */
	virtual EInferenceType get_inference_type() { return INF_LAPLACIAN; }

	/** returns the name of the inference method
	 *
	 * @return name Laplacian
	 */
	virtual const char* get_name() const { return "LaplacianInferenceMethod"; }

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
	virtual CMap<TParameter*, SGVector<float64_t> >
		get_marginal_likelihood_derivatives(CMap<TParameter*, CSGObject*>& para_dict);

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

	/** get tolerance for newton iterations
	 *
	 * @return tolerance for newton iterations
	 */
	virtual float64_t get_newton_tolerance() { return m_tolerance; }

	/** set tolerance for newton iterations
	 *
	 * @param tol tolerance for newton iterations to set
	 */
	virtual void set_newton_tolerance(float64_t tol) { m_tolerance=tol; }

	/** get max Newton iterations
	 *
	 * @return max Newton iterations
	 */
	virtual int32_t get_newton_iterations() { return m_iter; }

	/** set max Newton iterations
	 *
	 * @param iter max Newton iterations
	 */
	virtual void set_newton_iterations(int32_t iter) { m_iter=iter; }

	/** get tolerance for Brent's minimization method
	 *
	 * @return tolerance for Brent's minimization method
	 */
	virtual float64_t get_minimization_tolerance() { return m_opt_tolerance; }

	/** set tolerance for Brent's minimization method
	 *
	 * @param tol tolerance for Brent's minimization method
	 */
	virtual void set_minimization_tolerance(float64_t tol) { m_opt_tolerance=tol; }

	/** get max for Brent's minimization method
	 *
	 * @return max for Brent's minimization method
	 */
	virtual float64_t get_minimization_max() { return m_opt_max; }

	/** set max for Brent's minimization method
	 *
	 * @param iter max for Brent's minimization method
	 */
	virtual void set_minimization_max(float64_t max) { m_opt_max=max; }

	/**
	 * @return wether combination of Laplace approximation inference method and
	 * given likelihood function supports regression
	 */
	virtual bool supports_regression()
	{
		check_members();
		return m_model->supports_regression();
	}

	/**
	 * @return wether combination of Laplace approximation inference method and
	 * given likelihood function supports binary classification
	 */
	virtual bool supports_binary()
	{
		check_members();
		return m_model->supports_binary();
	}

	/** update data all matrices */
	virtual void update_all();

protected:
	/** update alpha matrix */
	virtual void update_alpha();

	/** update cholesky matrix */
	virtual void update_chol();

	/** update train kernel matrix */
	virtual void update_train_kernel();

private:
	void init();

private:
	/** Check if members of object are valid
	 * for inference
	 */
	void check_members();

	/** amount of tolerance for Newton's iterations */
	float64_t m_tolerance;

	/** max Newton's iterations */
	index_t m_iter;

	/** amount of tolerance for Brent's minimization method */
	float64_t m_opt_tolerance;

	/** max iterations for Brent's minimization method */
	float64_t m_opt_max;

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
