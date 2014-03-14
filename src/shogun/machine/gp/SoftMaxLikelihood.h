#ifndef _SOFTMAXLIKELIHOOD_H_
#define _SOFTMAXLIKELIHOOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/labels/MulticlassLabels.h>
#include <shogun/machine/gp/LikelihoodModel.h>

namespace shogun
{

/** @brief Class that models Soft-Max likelihood.
 *
 * softmax_i(f)=\frac{\exp{f_i}}{\sum\exp{f_i}}
 *
 */
class CSoftMaxLikelihood : public CLikelihoodModel
{
public:
	/** default constructor */
	CSoftMaxLikelihood();

	virtual ~CSoftMaxLikelihood();

	/** returns the name of the likelihood model
	 *
	 * @return name SoftMaxLikelihood
	 */
	virtual const char* get_name() const { return "SoftMaxLikelihood"; }

	/** returns mean of the predictive marginal \f$p(y_*|X,y,x_*)\f$
	 *
	 * NOTE: NOT IMPLEMENTED
	 *
	 * @param mu posterior mean of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param s2 posterior variance of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param lab labels \f$y_*\f$
	 *
	 * @return final means evaluated by likelihood function
	 */
	virtual SGVector<float64_t> get_predictive_means(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab=NULL) const
	{
		SG_ERROR("Not Implemented\n");
		return SGVector<float64_t>();
	}

	/** returns variance of the predictive marginal \f$p(y_*|X,y,x_*)\f$
	 *
	 * NOTE: NOT IMPLEMENTED
	 *
	 * @param mu posterior mean of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param s2 posterior variance of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param lab labels \f$y_*\f$
	 *
	 * @return final variances evaluated by likelihood function
	 */
	virtual SGVector<float64_t> get_predictive_variances(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab=NULL) const
	{
		SG_ERROR("Not Implemented\n");
		return SGVector<float64_t>();
	}

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

private:
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
#endif /* HAVE_EIGEN3 */
#endif /* _SOFTMAXLIKELIHOOD_H_ */
