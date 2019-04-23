/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Evan Shelhamer, Yuyu Zhang
 */

#ifndef _HISTOGRAM_H___
#define _HISTOGRAM_H___

#include <shogun/lib/config.h>

#include <shogun/features/StringFeatures.h>
#include <shogun/distributions/Distribution.h>

namespace shogun
{
	template <class ST> class StringFeatures;

/** @brief Class Histogram computes a histogram over all 16bit unsigned
 * integers in the features.
 *
 * Values in histogram are absolute counts (logarithmic)
 */
class Histogram : public Distribution
{
	public:
		/** default constructor */
		Histogram();

		/** constructor
		 *
		 * @param f histogram's features
		 */
		Histogram(std::shared_ptr<StringFeatures<uint16_t>> f);
		virtual ~Histogram();

		/** learn distribution
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(std::shared_ptr<Features> data=NULL);

		/** get number of model parameters
		 *
		 * @return number of model parameters
		 */
		virtual int32_t get_num_model_parameters() { return (1<<16); }

		/** get logarithm of given model parameter
		 *
		 * @param num_param which param
		 * @result logarithm of given model parameter
		 */
		virtual float64_t get_log_model_parameter(int32_t num_param);

		/** get logarithm of one example's derivative's likelihood
		 *
		 * @param num_param which example's param
		 * @param num_example which example
		 * @return logarithm of example's derivative's likelihood
		 */
		virtual float64_t get_log_derivative(
			int32_t num_param, int32_t num_example);

		/** get logarithm of one example's likelihood
		 *
		 * @param num_example which example
		 * @return logarithm of example's likelihood
		 */
		virtual float64_t get_log_likelihood_example(int32_t num_example);

		/** set histogram
		 *
		 * @param histogram new histogram
		 */
		virtual bool set_histogram(const SGVector<float64_t> histogram);

		/** get histogram
		 *
		 * @return current histogram
		 *
		 */
		virtual SGVector<float64_t> get_histogram();

		/** @return object name */
		virtual const char* get_name() const { return "Histogram"; }

	protected:
		/** histogram */
		float64_t* hist;
};
}
#endif
