/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 * Code adapted from
 * https://gist.github.com/yorkerlin/8a36e8f9b298aa0246a4
 * and
 * GPstuff - Gaussian process models for Bayesian analysis
 * http://becs.aalto.fi/en/research/bayes/gpstuff/
 *
 * The reference pseudo code is the algorithm 3.4 of the GPML textbook
 *
 */

#ifndef _SOFTMAXLIKELIHOOD_H_
#define _SOFTMAXLIKELIHOOD_H_

#include <shogun/lib/config.h>


#include <shogun/labels/MulticlassLabels.h>
#include <shogun/machine/gp/LikelihoodModel.h>

namespace shogun
{

/** mc sampler type */
enum EMCSamplerType
{
	MC_Probability,
	MC_Mean,
	MC_Variance
};

/** @brief Class that models Soft-Max likelihood.
 *
 * softmax_i(f)=\frac{\exp{f_i}}{\sum\exp{f_i}}
 *
 * Code adapted from
 * https://gist.github.com/yorkerlin/8a36e8f9b298aa0246a4
 * and
 * GPstuff - Gaussian process models for Bayesian analysis
 * http://becs.aalto.fi/en/research/bayes/gpstuff/
 *
 * The reference pseudo code is the algorithm 3.4 of the GPML textbook
 *
 * The implementation of predictive statistics is based on the mc sampler.
 * The basic idea of the sampler is that
 * first generating samples from the posterior Gaussian distribution given by mu and s2
 * and then using the samplers to estimate the predictive marginal distribution.
 *
 */
class CSoftMaxLikelihood : public CLikelihoodModel
{
public:
	/** default constructor */
	CSoftMaxLikelihood();

	/** destructor */
	virtual ~CSoftMaxLikelihood();

	/** returns the name of the likelihood model
	 *
	 * @return name SoftMaxLikelihood
	 */
	virtual const char* get_name() const { return "SoftMaxLikelihood"; }

	/** returns mean of the predictive marginal \f$p(y_*|X,y,x_*)\f$
	 * The implementation is based on a simple Monte Carlo sampler from the pseudo code.
	 *
	 * @param mu posterior mean of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param s2 posterior variance of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param lab labels \f$y_*\f$
	 *
	 * Note that the mean vector should be a column-marjor linearized C-by-n matrix,
	 * where C is the number of classes and n is the number of samplers
	 *
	 * @return final means (based on 0 and 1 bernoulli-encoding) evaluated by likelihood function
	 */
	virtual SGVector<float64_t> get_predictive_means(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab=NULL) const;

	/** returns variance of the predictive marginal \f$p(y_*|X,y,x_*)\f$
	 * The implementation is based on a simple Monte Carlo sampler from the pseudo code.
	 *
	 * @param mu posterior mean of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param s2 posterior variance of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param lab labels \f$y_*\f$
	 *
	 * Note that the variance vector should be a column-marjor linearized C-by-n matrix,
	 * where C is the number of classes and n is the number of samplers
	 *
	 * @return final variances (based on 0 and 1 bernoulli-encoding) evaluated by likelihood function
	 */
	virtual SGVector<float64_t> get_predictive_variances(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab=NULL) const;

	/** returns the logarithm of the predictive density of \f$y_*\f$:
	 * The implementation is based on a simple Monte Carlo sampler from the pseudo code.
	 *
	 * \f[
	 * log(p(y_*|X,y,x_*)) = log\left(\int p(y_*|f_*) p(f_*|X,y,x_*) df_*\right)
	 * \f]
	 *
	 * which approximately equals to
	 *
	 * \f[
	 * log\left(\int p(y_*|f_*) \mathcal{N}(f_*|\mu,\sigma^2) df_*\right)
	 * \f]
	 *
	 * where normal distribution \f$\mathcal{N}(\mu,\sigma^2)\f$ is an
	 * approximation to the posterior marginal \f$p(f_*|X,y,x_*)\f$.
	 *
	 * NOTE: if lab equals to NULL, then each \f$y_*\f$ equals to one.
	 *
	 * @param mu posterior mean of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param s2 posterior variance of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param lab labels \f$y_*\f$
	 *
	 * Note that the log_probability vector should be a column-marjor linearized C-by-n matrix,
	 * where C is the number of classes and n is the number of samplers
	 *
	 * @return \f$log(p(y_*|X, y, x*))\f$ for each label \f$y_*\f$ (based on 0 and 1 bernoulli-encoding)
	 */
	virtual SGVector<float64_t> get_predictive_log_probabilities(SGVector<float64_t> mu,
		SGVector<float64_t> s2, const CLabels *lab=NULL);

	/** returns the logarithm of the point-wise likelihood \f$log(p(y_i|f_i))\f$
	 * for each label \f$y_i\f$, an integer between 1 and C (ie. number of classes).
	 *
	 * One can evaluate log-likelihood like: \f$log(p(y|f)) = \sum_{i=1}^{n}
	 * log(p(y_i|f_i))\f$
	 *
	 * @param lab labels \f$y_i\f$, an integer between 1 and C (ie. num of classes)
	 * @param func values of the function \f$f_i\f$
	 *
	 * @return logarithm of the point-wise likelihood
	 */
	virtual SGVector<float64_t> get_log_probability_f(const CLabels* lab,
			SGVector<float64_t> func) const;

	/** get derivative of log likelihood \f$log(p(y|f))\f$ with respect to
	 * location function \f$f\f$
	 *
	 * @param lab labels \f$y_i\f$, an integer between 1 and C (ie. num of classes)
	 * @param func function location
	 * @param i index, choices are 1, 2, and 3 for first, second, and third
	 * derivatives respectively
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_log_probability_derivative_f(
			const CLabels* lab, SGVector<float64_t> func, index_t i) const;

	/** returns the zeroth moment of a given (unnormalized) probability
	 * distribution:
	 *
	 * NOTE: NOT IMPLEMENTED
	 *
	 * @param mu mean of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param s2 variance of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param lab labels \f$y_i\f$
	 *
	 * @return log zeroth moment \f$log(Z_i)\f$
	 */
	virtual SGVector<float64_t> get_log_zeroth_moments(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab) const
	{
		SG_ERROR("Not Implemented\n");
		return SGVector<float64_t>();
	}

	/** returns the first moment of a given (unnormalized) probability
	 * distribution \f$q(f_i) = Z_i^-1
	 * p(y_i|f_i)\mathcal{N}(f_i|\mu,\sigma^2)\f$, where \f$ Z_i=\int
	 * p(y_i|f_i)\mathcal{N}(f_i|\mu,\sigma^2) df_i\f$.
	 *
	 * NOTE: NOT IMPLEMENTED
	 *
	 * @param mu mean of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param s2 variance of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param lab labels \f$y_i\f$
	 * @param i index i
	 *
	 * @return first moment of \f$q(f_i)\f$
	 */
	virtual float64_t get_first_moment(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab, index_t i) const
	{
		SG_ERROR("Not Implemented\n");
		return -1.0;
	}

	/** returns the second moment of a given (unnormalized) probability
	 * distribution \f$q(f_i) = Z_i^-1
	 * p(y_i|f_i)\mathcal{N}(f_i|\mu,\sigma^2)\f$, where \f$ Z_i=\int
	 * p(y_i|f_i)\mathcal{N}(f_i|\mu,\sigma^2) df_i\f$.
	 *
	 * NOTE: NOT IMPLEMENTED
	 *
	 * @param mu mean of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param s2 variance of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param lab labels \f$y_i\f$
	 * @param i index i
	 *
	 * @return the second moment of \f$q(f_i)\f$
	 */
	virtual float64_t get_second_moment(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab, index_t i) const
	{
		SG_ERROR("Not Implemented\n");
		return -1.0;
	}

	/** return whether likelihood function supports multiclass classification
	 *
	 * @return true
	 */
	virtual bool supports_multiclass() const { return true; }

	/**
	 * set the num_samples used in the mc sampler
	 * @param num_samples number of samples to be generated
	 *
	 */
	virtual void set_num_samples(index_t num_samples);

private:
	/** init */
	void init();
	/** number of samples to be generated */
	index_t m_num_samples;

	/**use to get predictive statistics(probability,mean,variance)
	 *
	 * @param mu posterior mean of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param s2 posterior variance of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param lab labels \f$y_*\f$
	 * @param option predictive statistics(MC_Probability,MC_Mean,MC_Variance)
	 *
	 * @return the statistics based on mc sampler
	 */
	SGVector<float64_t> predictive_helper(SGVector<float64_t> mu,
	SGVector<float64_t> s2, const CLabels *lab, EMCSamplerType option) const;

	/**the Monte method sampler
	 *
	 * @param num_samples number of samples to be generated
	 * @param mu posterior mean of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param s2 posterior variance of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param y labels based on 0 and 1 encoding \f$y_*\f$
	 *
	 * @return the statistics based on mc sampler
	 */

	SGVector<float64_t> mc_sampler(index_t num_samples, SGVector<float64_t> mean,
		SGMatrix<float64_t> Sigma, SGVector<float64_t> y) const;


	/** get 1st derivative of log likelihood \f$log(p(y|f))\f$ with respect to
	 * location function \f$f\f$
	 *
	 * @param lab labels \f$y_i\f$, integers between 1 and C (ie. num of classes)
	 * @param func function (NxC where N is num vectors and C num classes)
	 *
	 * @return derivative (NxC matrix linearized in column major format)
	 */
	SGVector<float64_t> get_log_probability_derivative1_f(const CLabels* lab, SGMatrix<float64_t> func) const;

	/** get 2nd derivative of log likelihood \f$log(p(y|f))\f$ with respect to
	 * location function \f$f\f$
	 *
	 * @param func function (NxC where N is num vectors and C num classes)
	 *
	 * @return derivative (NCxC matrix [N blocks of CxC matrices concatenated along column]
	 * linearized in column major format)
	 */
	SGVector<float64_t> get_log_probability_derivative2_f(SGMatrix<float64_t> func) const;

	/** get 3rd derivative of log likelihood \f$log(p(y|f))\f$ with respect to
	 * location function \f$f\f$
	 *
	 * @param func function (NxC where N is num vectors and C num classes)
	 *
	 * @return derivative (NxCxCxC 4-d matrix linearized ie. Element(n,c1,c2,c3) =
	 * array[\f$n*C^{3}+c1*C^{2}+c2*C+c3\f$] where C is num_classes)
	 */
	SGVector<float64_t> get_log_probability_derivative3_f(SGMatrix<float64_t> func) const;
};
}
#endif /* _SOFTMAXLIKELIHOOD_H_ */
