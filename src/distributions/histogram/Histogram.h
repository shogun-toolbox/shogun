/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _HISTOGRAM_H___
#define _HISTOGRAM_H___

#include "features/StringFeatures.h"
#include "distributions/Distribution.h"

/** Class Histogram computes a histogram over all 16bit unsigned integers in the
 * features. Values in histogram are absolute counts (logarithmic) */
class CHistogram : public CDistribution
{
	public:
		/** default constructor */
		CHistogram();

		/** constructor
		 *
		 * @param f histogram's features
		 */
		CHistogram(CStringFeatures<WORD>* f);
		~CHistogram();

		/** train histogram
		 *
		 * @return if training was successful
		 */
		virtual bool train();

		/** get number of model parameters
		 *
		 * @return number of model parameters
		 */
		virtual inline INT get_num_model_parameters() { return (1<<16); }

		/** get logarithm of given model parameter
		 *
		 * @param num_param which param
		 * @result logarithm of given model parameter
		 */
		virtual DREAL get_log_model_parameter(INT num_param);

		/** get logarithm of one example's derivative's likelihood
		 *
		 * @param num_param which example's param
		 * @param num_example which example
		 * @return logarithm of example's derivative's likelihood
		 */
		virtual DREAL get_log_derivative(INT num_param, INT num_example);

		/** get logarithm of one example's likelihood
		 *
		 * @param num_example which example
		 * @return logarithm of example's likelihood
		 */
		virtual DREAL get_log_likelihood_example(INT num_example);

		/** set histogram
		 *
		 * @param src new histogram
		 * @param num number of values in histogram
		 */
		virtual bool set_histogram(DREAL* src, INT num);

		/** get histogram
		 *
		 * @param dst where the histogram will be stored
		 * @param num where number of values in histogram will be
		 *        stored
		 */
		virtual void get_histogram(DREAL** dst, INT* num);

	protected:
		/** histogram */
		DREAL* hist;
};
#endif
