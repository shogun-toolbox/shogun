/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Bjoern Esser, Thoralf Klein, Fernando Iglesias,
 *          Yuyu Zhang
 */

#ifndef _DISTRIBUTION_H___
#define _DISTRIBUTION_H___

#include <shogun/lib/config.h>

#include <shogun/features/Features.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/SGObject.h>

namespace shogun
{
class Features;
class Math;
/** @brief Base class Distribution from which all methods implementing a
 * distribution are derived.
 *
 * Distributions are based on some general feature object and have to implement
 * interfaces to
 *
 * train()						- for learning a distribution
 * get_num_model_parameters()	- for the total number of model parameters
 * get_log_model_parameter()	- for the n-th model parameter (logarithmic)
 * get_log_derivative()			- for the partial derivative wrt. to the n-th
 *										model parameter
 * get_log_likelihood_example() - for the likelihood for the
 *										n-th example
 *
 * This way methods building on Distribution, might enumerate over all possible
 * model parameters and obtain the parameter vector and the gradient. This is
 * used to compute e.g. the TOP and Fisher Kernel (cf. PluginEstimate, CHistogramKernel,
 * CTOPFeatures and CFKFeatures ).
 */
class Distribution : public SGObject
{
	public:
		/** default constructor */
		Distribution();

		/** destructor */
		virtual ~Distribution();

		/** learn distribution
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(std::shared_ptr<Features> data=NULL)=0;

		/** get number of parameters in model
		 *
		 * abstract base method
		 *
		 * @return number of parameters in model
		 */
		virtual int32_t get_num_model_parameters()=0;

		/** get number of parameters in model that are relevant,
		 * i.e. > ALMOST_NEG_INFTY
		 *
		 * @return number of relevant model parameters
		 */
		virtual int32_t get_num_relevant_model_parameters();

		/** get model parameter (logarithmic)
		 *
		 * abstract base method
		 *
		 * @return model parameter (logarithmic)
		 */
		virtual float64_t get_log_model_parameter(int32_t num_param)=0;

		/** get partial derivative of likelihood function (logarithmic)
		 *
		 * abstract base method
		 *
		 * @param num_param derivative against which param
		 * @param num_example which example
		 * @return derivative of likelihood (logarithmic)
		 */
		virtual float64_t get_log_derivative(
			int32_t num_param, int32_t num_example)=0;

		/** compute log likelihood for example
		 *
		 * abstract base method
		 *
		 * @param num_example which example
		 * @return log likelihood for example
		 */
		virtual float64_t get_log_likelihood_example(int32_t num_example)=0;

		/** compute log likelihood for whole sample
		 *
		 * @return log likelihood for whole sample
		 */
		virtual float64_t get_log_likelihood_sample();

		/** compute log likelihood for each example
		 *
		 * @return log likelihood vector
		 */
		virtual SGVector<float64_t> get_log_likelihood();

		/** get model parameter
		 *
		 * @param num_param which param
		 * @return model parameter
		 */
		virtual float64_t get_model_parameter(int32_t num_param)
		{
			return exp(get_log_model_parameter(num_param));
		}

		/** get partial derivative of likelihood function
		 *
		 * @param num_param partial derivative against which param
		 * @param num_example which example
		 * @return derivative of likelihood function
		 */
		virtual float64_t get_derivative(
			int32_t num_param, int32_t num_example)
		{
			return exp(get_log_derivative(num_param, num_example));
		}

		/** compute likelihood for example
		 *
		 * @param num_example which example
		 * @return likelihood for example
		 */
		virtual float64_t get_likelihood_example(int32_t num_example)
		{
			return exp(get_log_likelihood_example(num_example));
		}

		/** compute likelihood for all vectors in sample
		 *
		 * @return likelihood vector for all examples
		 */
		virtual SGVector<float64_t> get_likelihood_for_all_examples();

		/** set feature vectors
		 *
		 * @param f new feature vectors
		 */
		virtual void set_features(std::shared_ptr<Features> f)
		{
			features=f;
		}

		/** get feature vectors
		 *
		 * @return feature vectors
		 */
		virtual std::shared_ptr<Features> get_features()
		{
			return features;
		}

		/** set pseudo count
		 *
		 * @param pseudo new pseudo count
		 */
		virtual void set_pseudo_count(float64_t pseudo) { pseudo_count=pseudo; }

		/** get pseudo count
		 *
		 * @return pseudo count
		 */
		virtual float64_t get_pseudo_count() { return pseudo_count; }

		/** update parameters in the em maximization step for mixture model of which
		 * this distribution is a part
		 *
		 * abstract base method
		 *
		 * @param alpha_k "belongingness" values of various data points
		 * @return sum of alpha_k values
		 */
		virtual float64_t update_params_em(const SGVector<float64_t> alpha_k);

		/** obtain from generic
		 *
		 * @param object generic object
		 * @return Distribution object
		 */
#ifndef SWIG
		[[deprecated("use .as template function")]]
#endif
		static std::shared_ptr<Distribution> obtain_from_generic(const std::shared_ptr<SGObject>& object);

	protected:
		/** feature vectors */
		std::shared_ptr<Features> features;
		/** pseudo count */
		float64_t pseudo_count;
};
}
#endif
