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

#ifndef CINFERENCEMETHOD_H_
#define CINFERENCEMETHOD_H_

#include <shogun/kernel/Kernel.h>
#include <shogun/base/SGObject.h>
#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>
#include <shogun/machine/gp/LikelihoodModel.h>
#include <shogun/machine/gp/MeanFunction.h>
#include <shogun/evaluation/DifferentiableFunction.h>

namespace shogun
{

/** inference type */
enum EInferenceType
{
	INF_NONE = 0,
	INF_EXACT = 10,
	INF_FITC = 20,
	INF_LAPLACIAN = 30
};

/** @brief The Inference Method base class.
 *
 * The Inference Method computes (approximately) the posterior
 * distribution for a given Gaussian Process.
 */
class CInferenceMethod : public CDifferentiableFunction
{
public:
	/** default constructor */
	CInferenceMethod();

	/** constructor
	 *
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param mean mean function
	 * @param labels labels of the features
	 * @param model likelihood model to use
	 */
	CInferenceMethod(CKernel* kernel, CFeatures* features,
		CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

	virtual ~CInferenceMethod();

	/** return what type of inference we are, e.g. exact, FITC,
	 * Laplacian, etc.
	 *
	 * @return inference type
	 */
	virtual EInferenceType get_inference_type() { return INF_NONE; }

	/** get negative log marginal likelihood
	 *
	 * @return the negative log of the marginal likelihood function:
	 *
	 * \f[
	 * -log(p(y|X, \theta))
	 * \f]
	 *
	 * where \f$y\f$ are the labels, \f$X\f$ are the features, and
	 * \f$\theta\f$ represent hyperparameters.
	 */
	virtual float64_t get_negative_marginal_likelihood()=0;

	/** get log marginal likelihood gradient
	 *
	 * @return vector of the marginal likelihood function gradient
	 * with respect to hyperparameters:
	 *
	 * \f[
	 * -\frac{\partial log(p(y|X, \theta))}{\partial \theta}
	 * \f]
	 *
	 * where \f$y\f$ are the labels, \f$X\f$ are the features, and
	 * \f$\theta\f$ represent hyperparameters.
	 */
	virtual CMap<TParameter*, SGVector<float64_t> >	get_marginal_likelihood_derivatives(
		CMap<TParameter*, CSGObject*>& para_dict)=0;

	/** get alpha vector
	 *
	 * @return vector to compute posterior mean of Gaussian Process:
	 *
	 * \f[
	 * \mu = K\alpha
	 * \f]
	 *
	 * where \f$\mu\f$ is the mean and \f$K\f$ is the prior covariance
	 * matrix.
	 */
	virtual SGVector<float64_t> get_alpha()=0;

	/** get Cholesky decomposition matrix
	 *
	 * @return Cholesky decomposition of matrix:
	 *
	 * \f[
	 * L = cholesky(sW*K*sW+I)
	 * \f]
	 *
	 * where \f$K\f$ is the prior covariance matrix, \f$sW\f$ is the
	 * vector returned by get_diagonal_vector(), and \f$I\f$ is the
	 * identity matrix.
	 */
	virtual SGMatrix<float64_t> get_cholesky()=0;

	/** get diagonal vector
	 *
	 * @return diagonal of matrix used to calculate posterior
	 * covariance matrix:
	 *
	 * \f[
	 * Cov = (K^{-1}+sW^{2})^{-1}
	 * \f]
	 *
	 * where \f$Cov\f$ is the posterior covariance matrix, \f$K\f$ is
	 * the prior covariance matrix, and \f$sW\f$ is the diagonal
	 * vector.
	 */
	virtual SGVector<float64_t> get_diagonal_vector()=0;

	/** get the gradient
	 *
	 * @return map of gradient: keys are names of parameters, values
	 * are values of derivative with respect to that parameter.
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
		result[0]=get_negative_marginal_likelihood();
		return result;
	}

	/** returns mean vector \f$\mu\f$ of the Gaussian distribution
	 * \f$N(\mu,\Sigma)\f$, which is an approximation to the
	 * posterior:
	 *
	 * \f[
	 * p(f|y) \approx q(f|y) = N(\mu,\Sigma)
	 * \f]
	 *
	 * @return mean vector
	 */
	virtual SGVector<float64_t> get_posterior_approximation_mean()
	{
		SG_ERROR("Inference method doesn't use approximation to the posterior")
		return SGVector<float64_t>();
	}

	/** returns covariance matrix \f$\Sigma\f$ of the Gaussian
	 * distribution \f$q(f|y)=N(\mu,\Sigma)\f$, which is an
	 * approximation to the posterior:
	 *
	 * \f[
	 * p(f|y) \approx q(f|y) = N(\mu,\Sigma)
	 * \f]
	 *
	 * @return covariance matrix
	 */
	virtual SGMatrix<float64_t> get_posterior_approximation_covariance()
	{
		SG_ERROR("Inference method doesn't use approximation to the posterior")
		return SGMatrix<float64_t>();
	}

	/** set features
	*
	* @param feat features to set
	*/
	virtual void set_features(CFeatures* feat);

	/** get features
	*
	* @return features
	*/
	virtual CFeatures* get_features() {	SG_REF(m_features);	return m_features; }

	/** get kernel
	 *
	 * @return kernel
	 */
	virtual CKernel* get_kernel() { SG_REF(m_kernel); return m_kernel; }

	/** set kernel
	 *
	 * @param kern kernel to set
	 */
	virtual void set_kernel(CKernel* kern);

	/** get mean
	 *
	 * @return mean
	 */
	virtual CMeanFunction* get_mean() { SG_REF(m_mean); return m_mean; }

	/** set mean
	 *
	 * @param m mean function to set
	 */
	virtual void set_mean(CMeanFunction* m);

	/** get labels
	 *
	 * @return labels
	 */
	virtual CLabels* get_labels() { SG_REF(m_labels); return m_labels; }

	/** set labels
	 *
	 * @param lab label to set
	 */
	virtual void set_labels(CLabels* lab);

	/** get likelihood model
	 *
	 * @return likelihood
	 */
	CLikelihoodModel* get_model() { SG_REF(m_model); return m_model; }

	/** set likelihood model
	 *
	 * @param mod model to set
	 */
	virtual void set_model(CLikelihoodModel* mod);

	/** set kernel scale
	 *
	 * @param s scale to be set
	 */
	virtual void set_scale(float64_t s);

	/** get kernel scale
	 *
	 * @return kernel scale
	 */
	virtual float64_t get_scale() { return m_scale; }

	/** whether combination of inference method and given likelihood
	 * function supports regression
	 *
	 * @return false
	 */
	virtual bool supports_regression() { return false; }

	/** whether combination of inference method and given likelihood
	 * function supports binary classification
	 *
	 * @return false
	 */
	virtual bool supports_binary() { return false; }

	/** whether combination of inference method and given likelihood
	 * function supports multiclass classification
	 *
	 * @return false
	 */
	virtual bool supports_multiclass() { return false; }

	/** update all matrices */
	virtual void update_all()=0;

protected:
	/** update alpha matrix */
	virtual void update_alpha()=0;

	/** update cholesky matrix */
	virtual void update_chol()=0;

	/** update train kernel matrix */
	virtual void update_train_kernel()=0;

private:
	void init();

protected:
	/** covariance function */
	CKernel* m_kernel;

	/** mean function */
	CMeanFunction* m_mean;

	/** likelihood function to use */
	CLikelihoodModel* m_model;

	/** features to use */
	CFeatures* m_features;

	/** labels of features */
	CLabels* m_labels;

	/** feature matrix */
	SGMatrix<float64_t> m_feature_matrix;

	/** alpha vector used in process mean calculation */
	SGVector<float64_t> m_alpha;

	/** upper triangular factor of Cholesky decomposition */
	SGMatrix<float64_t> m_L;

	/** kernel scale */
	float64_t m_scale;

	/** kernel matrix from features */
	SGMatrix<float64_t> m_ktrtr;
};
}
#endif /* CINFERENCEMETHOD_H_ */
