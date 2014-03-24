/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CGHMM_H___
#define _CGHMM_H___

#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/Features.h>
#include <shogun/distributions/Distribution.h>

namespace shogun
{
/** @brief class GHMM - this class is non-functional and was meant to implement a
 * Generalize Hidden Markov Model (aka Semi Hidden Markov HMM). */
class CGHMM : public CDistribution
{
	public:
		/** default constructor */
		CGHMM();
		virtual ~CGHMM();

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
		virtual int32_t get_num_model_parameters();

		/** get logarithm of given model parameter
		 *
		 * @param param_num which param
		 * @result logarithm of given model parameter
		 */
		virtual float64_t get_log_model_parameter(int32_t param_num);

		/** get logarithm of one example's derivative's likelihood
		 *
		 * @param param_num which example's param
		 * @param num_example which example
		 * @return logarithm of example's derivative's likelihood
		 */
		virtual float64_t get_log_derivative(int32_t param_num, int32_t num_example);

		/** get logarithm of one example's likelihood
		 *
		 * @param num_example which example
		 * @return logarithm of example's likelihood
		 */
		virtual float64_t get_log_likelihood_example(int32_t num_example);

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 * @return name of the SGSerializable
		 */
	virtual const char* get_name() const { return "GHMM"; }
};
}
#endif
