/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _HISTOGRAM_H___
#define _HISTOGRAM_H___

#include "features/StringFeatures.h"
#include "distributions/Distribution.h"

class CHistogram : public CDistribution
{
	public:
		CHistogram();
		CHistogram(CStringFeatures<WORD>* f);
		~CHistogram();

		virtual bool train();

		virtual inline INT get_num_model_parameters() { return (1<<16); }
		virtual DREAL get_log_model_parameter(INT num_param);

		virtual DREAL get_log_derivative(INT num_param, INT num_example);
		virtual DREAL get_log_likelihood_example(INT num_example);

		virtual bool set_histogram(DREAL* src, INT num);
		virtual void get_histogram(DREAL** dst, INT* num);

	protected:
		DREAL* hist;
};
#endif
