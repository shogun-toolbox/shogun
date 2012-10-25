/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _HISTOGRAM_H___
#define _HISTOGRAM_H___

#include <shogun/features/StringFeatures.h>
#include <shogun/distributions/Distribution.h>

namespace shogun
{
	template <class ST> class CStringFeatures;

/** @brief Class Histogram computes a histogram over all 16bit unsigned
 * integers in the features.
 *
 * Values in histogram are absolute counts (logarithmic)
 */
class CHistogram : public CDistribution
{
	public:
		/** default constructor */
		CHistogram();

		/** constructor
		 *
		 * @param f histogram's features
		 */
		CHistogram(CStringFeatures<uint16_t>* f);
		virtual ~CHistogram();

		/** learn distribution
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

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
