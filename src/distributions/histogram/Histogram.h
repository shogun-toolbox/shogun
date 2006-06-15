/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _HISTOGRAM_H___
#define _HISTOGRAM_H___

#include "features/WordFeatures.h"
#include "distributions/Distribution.h"

class CHistogram : private CDistribution
{
	public:
		CHistogram();
		~CHistogram();

		virtual bool train();

		virtual inline INT get_num_model_parameters()
		{
			return (1<<16);
		}
		virtual DREAL get_log_model_parameter(INT param_num);

		virtual DREAL get_log_derivative(INT num_example, INT num_param);
		virtual DREAL get_log_likelihood_example(INT num_example);

	protected:
		DREAL* hist;
		CWordFeatures* features;
};
#endif
