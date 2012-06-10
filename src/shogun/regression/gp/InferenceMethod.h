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

#include <shogun/kernel/Kernel.h>
#include <shogun/base/SGObject.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/labels/Labels.h>
#include <shogun/regression/gp/LikelihoodModel.h>
#include <shogun/regression/gp/MeanFunction.h>

namespace shogun {

/** @brief The Inference Method base class.
 *
 *  The Inference Method computes approximately the
 *  posterior distribution for a given Gaussian Process.
 *
 */
class CInferenceMethod : public CSGObject {
public:

	/* Default Constructor
	 *
	 */
	CInferenceMethod();

	/* Constructor
	 * @param kernel covariance function
	 * @param features features to use in inference
	 * @param labels labels of the features
	 * @param model Likelihood model to use
	 */
	CInferenceMethod(CKernel* kernel, CDotFeatures* features,
			CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model);

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
	virtual SGVector<float64_t> get_marginal_likelihood_derivatives() = 0;

	/** get Alpha Matrix
	 *
	 * @return Matrix to compute posterior mean of Gaussian Process:
	 * \f[
	 *		\mu = K\alpha
	 * \f]
	 *
	 * 	where \mu is the mean and K is the prior covariance matrix
	 */
	virtual SGVector<float64_t> get_alpha() = 0;

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
	virtual inline void set_features(CDotFeatures* feat)
	{
		SG_UNREF(features);
		SG_REF(feat);
		features=feat;
	}

	/** get features
	*
	* @return features
	*/
	virtual CDotFeatures* get_features() { SG_REF(features); return features; }

	/**get kernel
	 *
	 * @return kernel
	 */
	virtual CKernel* get_kernel() { SG_REF(kernel); return kernel; }

	/**set kernel
	 *
	 * @param kern kernel to set
	 */
	virtual inline void set_kernel(CKernel* kern)
	{
		SG_UNREF(kernel);
		SG_REF(kern);
		kernel=kern;
	}

	/**get kernel
	 *
	 * @return kernel
	 */
	virtual CMeanFunction* get_mean() { SG_REF(mean); return mean; }

	/**set kernel
	 *
	 * @param kern kernel to set
	 */
	virtual inline void set_mean(CMeanFunction* m)
	{
		SG_UNREF(mean);
		SG_REF(m);
		mean=m;
	}

	/**get labels
	 *
	 * @return labels
	 */
	virtual CLabels* get_labels() { SG_REF(m_labels); return m_labels; }

	/**set labels
	 *
	 * @param lab label to set
	 */
	virtual inline void set_labels(CLabels* lab)
	{
		SG_UNREF(m_labels);
		SG_REF(lab);
		m_labels=lab;
	}

	/**get likelihood model
	 *
	 * @return likelihood
	 */
	CLikelihoodModel* get_model() {SG_REF(m_model); return m_model; }

	/**set likelihood model
	 *
	 * @param mod model to set
	 */
	inline void set_model(CLikelihoodModel* mod)
	{
		SG_UNREF(m_model);
		SG_REF(mod);
		m_model=mod;
	}

protected:

	/*Covariance Function*/
	CKernel* kernel;

	/*Features to use*/
	CDotFeatures* features;

	/*Labels of those features*/
	CLabels* m_labels;

	/*Mean Function*/
	CMeanFunction* mean;

	/*Likelihood function to use
	 * \f[
	 *   p(y|f)
	 * \f]
	 *
	 * Where y are the labels and f is the prediction
	 * function
	 *
	 */
	CLikelihoodModel* m_model;

	/* alpha matrix used in process mean calculation */
	SGVector< float64_t > m_alpha;

	/** Lower triangle Cholesky decomposition of
	 *  feature matrix
	 */
	SGMatrix<float64_t> m_L;
};

}

#endif /* CInferenceMethod_H_ */
