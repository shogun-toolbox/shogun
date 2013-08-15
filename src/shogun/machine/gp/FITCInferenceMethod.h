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

#include <shogun/machine/gp/InferenceMethod.h>

namespace shogun
{

/** @brief The Fully Independent Conditional Training inference method class.
 *
 * This inference method computes the Cholesky and Alpha vectors approximately
 * with the help of latent variables. For more details, see "Sparse Gaussian
 * Process using Pseudo-inputs", Edward Snelson, Zoubin Ghahramani, NIPS 18, MIT
 * Press, 2005.
 *
 * This specific implementation was inspired by the infFITC.m file in the GPML
 * toolbox.
 *
 * NOTE: The Gaussian Likelihood Function must be used for this inference
 * method.
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
	virtual EInferenceType get_inference_type() const { return INF_FITC; }

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
	virtual void set_latent_features(CFeatures* feat)
	{
		SG_REF(feat);
		SG_UNREF(m_latent_features);
		m_latent_features=feat;
	}

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
	 * where \f$y\f$ are the labels, \f$X\f$ are the features, and \f$\theta\f$
	 * represent hyperparameters.
	 */
	virtual float64_t get_negative_marginal_likelihood();

	/** get log marginal likelihood gradient
	 *
	 * @return vector of the marginal likelihood function gradient with respect
	 * to hyperparameters:
	 *
	 * \f[
	 *	 -\frac{\partial {log(p(y|X, \theta))}}{\partial \theta}
	 * \f]
	 *
	 * where \f$y\f$ are the labels, \f$X\f$ are the features, and \f$\theta\f$
	 * represent hyperparameters.
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
	 * where \f$K\f$ is the prior covariance matrix, \f$sW\f$ is the vector
	 * returned by get_diagonal_vector(), and \f$I\f$ is the identity matrix.
	 */
	virtual SGMatrix<float64_t> get_cholesky();

	/** get diagonal vector
	 *
	 * @return diagonal of matrix used to calculate posterior covariance matrix:
	 *
	 * \f[
	 *	    Cov = (K^{-1}+sW^{2})^{-1}
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

	/** update all matrices */
	virtual void update();

protected:
	/** check if members of object are valid for inference */
	virtual void check_members() const;

	/** update alpha matrix */
	virtual void update_alpha();

	/** update cholesky Matrix.*/
	virtual void update_chol();

	/** update train kernel matrix */
	virtual void update_train_kernel();

private:
	void init();

private:
	/** latent features for approximation */
	CFeatures* m_latent_features;

	/** noise of the latent variables */
	float64_t m_ind_noise;

	/** Cholesky of covariance of latent features */
	SGMatrix<float64_t> m_chol_uu;

	/** Cholesky of covariance of latent features and training features */
	SGMatrix<float64_t> m_chol_utr;

	/** covariance matrix of latent features */
	SGMatrix<float64_t> m_kuu;

	/** covariance matrix of latent features and training features */
	SGMatrix<float64_t> m_ktru;

	/** diagonal of training kernel matrix + noise - diagonal of the matrix
	 * (m_chol_uu^{-1}*m_ktru)* (m_chol_uu^(-1)*m_ktru)' = V*V'
	 */
	SGVector<float64_t> m_dg;

	/** labels adjusted for noise and means */
	SGVector<float64_t> m_r;

	/** solves the equation V * r = m_chol_utr */
	SGVector<float64_t> m_be;
};
}
#endif // HAVE_EIGEN3
#endif /* CFITCINFERENCEMETHOD_H_ */
