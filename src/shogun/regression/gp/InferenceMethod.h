/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CINFERENCEMETHOD_H_
#define CINFERENCEMETHOD_H_
#ifdef HAVE_EIGEN3
#include <shogun/kernel/Kernel.h>
#include <shogun/base/SGObject.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/labels/Labels.h>
#include <shogun/regression/gp/LikelihoodModel.h>
#include <shogun/regression/gp/MeanFunction.h>
#include <shogun/evaluation/DifferentiableFunction.h>
#include <shogun/labels/RegressionLabels.h>


namespace shogun
{

/** @brief The Inference Method base class.
 *
 *  The Inference Method computes approximately the
 *  posterior distribution for a given Gaussian Process.
 *
 */
class CInferenceMethod : public CDifferentiableFunction
{
  
public:

	/** Default Constructor
	 *
	 */
	CInferenceMethod();

	/** Constructor
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean Mean function
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	CInferenceMethod(CKernel* kernel, CFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

	/** destructor */
	virtual ~CInferenceMethod();

	/** get Negative Log Marginal Likelihood
	 *
	 * @return The Negative Log of the Marginal Likelihood function:
	 * \f[
	 *	  -log(p(y|X, \theta))
	 *	  Where y are the labels, X are the features,
	 *	  and \theta represent hyperparameters
	 * \f]
	 */
	virtual float64_t get_negative_marginal_likelihood() = 0;

	/** get Log Marginal Likelihood Gradient
	 *
	 * @return Vector of the  Marginal Likelihood Function Gradient
	 *         with respect to hyperparameters
	 * \f[
	 *	 -\frac{\partial {log(p(y|X, \theta))}}{\partial \theta}
	 * \f]
	 */
	virtual CMap<TParameter*, SGVector<float64_t> >
		get_marginal_likelihood_derivatives(
		CMap<TParameter*, CSGObject*>& para_dict) = 0;

	/** get Alpha Matrix
	 *
	 * @return Matrix to compute posterior mean of Gaussian Process:
	 * \f[
	 *		\mu = K\alpha
	 * \f]
	 *
	 * 	where \f$\mu\f$ is the mean and \f$K\f$ is the prior covariance matrix
	 */
	virtual SGVector<float64_t> get_alpha() = 0;

	/** get Cholesky Decomposition Matrix
	 *
	 * @return Cholesky Decomposition of Matrix:
	 * \f[
	 *		 L = Cholesky(sW*K*sW+I)
	 * \f]
	 *
	 * 	Where \f$K\f$ is the prior covariance matrix, sW is the matrix returned by
	 * 	get_cholesky(), and \f$I\f$ is the identity matrix.
	 */
	virtual SGMatrix<float64_t> get_cholesky() = 0;

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
	virtual SGVector<float64_t> get_diagonal_vector() = 0;

	/** set features
	*
	* @param feat features to set
	*/
	virtual void set_features(CFeatures* feat);

	/** get features
	*
	* @return features
	*/
	virtual CFeatures* get_features()
	{
		SG_REF(m_features);
		return m_features;
	}

	/**get kernel
	 *
	 * @return kernel
	 */
	virtual CKernel* get_kernel() { SG_REF(m_kernel); return m_kernel; }

	/**set kernel
	 *
	 * @param kern kernel to set
	 */
	virtual void set_kernel(CKernel* kern);

	/**get mean
	 *
	 * @return mean
	 */
	virtual CMeanFunction* get_mean() { SG_REF(m_mean); return m_mean; }

	/**set mean
	 *
	 * @param m mean function to set
	 */
	virtual void set_mean(CMeanFunction* m);

	/**get labels
	 *
	 * @return labels
	 */
	virtual CLabels* get_labels() { SG_REF(m_labels); return m_labels; }

	/**set labels
	 *
	 * @param lab label to set
	 */
	virtual void set_labels(CLabels* lab);

	/**get likelihood model
	 *
	 * @return likelihood
	 */
	CLikelihoodModel* get_model() {SG_REF(m_model); return m_model; }

	/**set likelihood model
	 *
	 * @param mod model to set
	 */
	virtual void set_model(CLikelihoodModel* mod);

	/**set kernel scale
	 *
	 * @param s scale to be set
	 */
	virtual void set_scale(float64_t s);

	/**get kernel scale
	 *
	 * @return kernel scale
	 */
	virtual float64_t get_scale() { return m_scale; }

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


protected:

	/** Update alpha matrix */
	virtual void update_alpha() {}

	/** Update cholesky matrix */
	virtual void update_chol() {}

	/** Update train kernel matrix */
	virtual void update_train_kernel() {}

	/** Update data means */
	virtual void update_data_means();

private:
	void init();

protected:

	/**Covariance Function*/
	CKernel* m_kernel;

	/**Features to use*/
	CFeatures* m_features;

	/**Feature Matrix*/
	SGMatrix<float64_t> m_feature_matrix;

	/**Means of Features*/
	SGVector<float64_t> m_data_means;

	/**Vector of labels*/
	SGVector<float64_t> m_label_vector;


	/**Labels of those features*/
	CLabels* m_labels;

	/**Mean Function*/
	CMeanFunction* m_mean;

	/**Latent Features for Approximation*/
	CFeatures* m_latent_features;

	/**Likelihood function to use
	 * \f[
	 *   p(y|f)
	 * \f]
	 *
	 * Where y are the labels and f is the prediction
	 * function
	 *
	 */
	CLikelihoodModel* m_model;

	/** alpha matrix used in process mean calculation */
	SGVector< float64_t > m_alpha;

	/** Lower triangle Cholesky decomposition of
	 *  feature matrix
	 */
	SGMatrix<float64_t> m_L;

	/**Kernel Scale*/
	float64_t m_scale;

	/**Kernel matrix from features*/
	SGMatrix<float64_t> m_ktrtr;

	/** Kernel matrix from latent features */
	SGMatrix<float64_t> m_latent_matrix;

};

}
#endif /* HAVE_EIGEN3 */
#endif /* CInferenceMethod_H_ */
