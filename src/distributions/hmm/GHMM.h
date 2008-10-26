/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CGHMM_H___
#define _CGHMM_H___

#include "lib/Mathematics.h"
#include "features/Features.h"
#include "distributions/Distribution.h"


/** class GHMM - this class is non-functional and was meant to implement a
 * Generalize Hidden Markov Model (aka Semi Hidden Markov HMM) */
class CGHMM : public CDistribution
{
	public:
		/** default constructor */
		CGHMM();
		~CGHMM();

		/** train distribution
		 *
		 * @return if training was successful
		 */
		virtual bool train();

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
		virtual DREAL get_log_model_parameter(int32_t param_num);

		/** get logarithm of one example's derivative's likelihood
		 *
		 * @param param_num which example's param
		 * @param num_example which example
		 * @return logarithm of example's derivative's likelihood
		 */
		virtual DREAL get_log_derivative(int32_t param_num, int32_t num_example);

		/** get logarithm of one example's likelihood
		 *
		 * @param num_example which example
		 * @return logarithm of example's likelihood
		 */
		virtual DREAL get_log_likelihood_example(int32_t num_example);
};
#endif
