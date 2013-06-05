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

#ifndef CFITCINFERENCEMETHOD_H_
#define CFITCINFERENCEMETHOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/regression/gp/InferenceMethod.h>

namespace shogun
{

/** @brief The Fully Independent Conditional Training Inference Method.
 *
 *  This inference method computes the Cholesky and
 *  Alpha vectors approximately with the help of latent
 *  variables. For more details, see "Sparse Gaussian Process
 *  using Pseudo-inputs", Edward Snelson, Zoubin Ghahramani,
 *  NIPS 18, MIT Press, 2005.
 *
 *  This specific implementation was inspired by the infFITC.m file
 *  in the GPML toolbox
 *
 *  The Gaussian Likelihood Function must be used for this inference method.
 */
class CFITCInferenceMethod: public CInferenceMethod
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
	 * @param latent_features features to use
	 */
	CFITCInferenceMethod(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model,
			CFeatures* latent_features);

	virtual ~CFITCInferenceMethod();

	/** return what type of inference we are
	 *
	 * @return inference type FITC
	 */
	virtual EInferenceType get_inference_type() { return INF_FITC; }

	/** returns the name of the inference method
	 *
	 * @return name FITC
	 */
	virtual const char* get_name() const { return "FITCInferenceMethod"; }

	/** helper method used to specialize a base class instance
	 *
	 * @param inference inference method
	 * @return casted CFITCInferenceMethod object
	 */
	static CFITCInferenceMethod* obtain_from_generic(CInferenceMethod* inference);

	/** set latent features
	 *
	 * @param feat features to set
	 */
	virtual void set_latent_features(CFeatures* feat);

	/** get latent features
	 *
	 * @return features
	 */
	virtual CFeatures* get_latent_features()
	{
		SG_REF(m_latent_features);
		return m_latent_features;
	}

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

	/** latent features for approximation */
	CFeatures* m_latent_features;

	/** kernel matrix from latent features */
	SGMatrix<float64_t> m_latent_matrix;

	/*noise of the latent variables*/
	float64_t m_ind_noise;

	/*Cholesky of Covariance of
	 * latent features
	 */
	SGMatrix<float64_t> m_chol_uu;

	/*Cholesky of Covariance of
	 * latent features
	 * and training features
	 */
	SGMatrix<float64_t> m_chol_utr;

	/* Covariance matrix of latent
	 * features
	 */
	SGMatrix<float64_t> m_kuu;

	/* Covariance matrix of latent
	 * features and training features
	 */
	SGMatrix<float64_t> m_ktru;

	/* Diagonal of Training
	 * kernel matrix + noise
	 * - diagonal of the matrix
	 * (m_chol_uu^{-1}*m_ktru)*
	 * (m_chol_uu^(-1)*m_ktru)'
	 * = V*V'
	 */
	SGVector<float64_t> m_dg;

	/*Labels adjusted for
	 * noise and means
	 */
	SGVector<float64_t> m_r;

	/* Solves the equation
	 * V*r = m_chol_utr
	 */
	SGVector<float64_t> m_be;
};
}
#endif // HAVE_EIGEN3
#endif /* CFITCINFERENCEMETHOD_H_ */
