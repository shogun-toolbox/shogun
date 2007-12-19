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

class CHistogram : private CDistribution
{
	public:
		CHistogram();
		CHistogram(CStringFeatures<WORD>* f);
		~CHistogram();

		virtual bool train();

		virtual inline INT get_num_model_parameters()
		{
			return (1<<16);
		}
		virtual DREAL get_log_model_parameter(INT num_param);

		virtual DREAL get_log_derivative(INT num_example, INT num_param);
		virtual DREAL get_log_likelihood_example(INT num_example);

		virtual inline bool set_histogram(DREAL* src, INT num)
		{
			ASSERT(num==get_num_model_parameters());

			delete[] hist;
			hist=new DREAL[num];
			ASSERT(hist);

			for (INT i=0; i<num; i++) {
				hist[i]=src[i];
			}

			return true;
		}

		virtual inline void get_histogram(DREAL** dst, INT* num)
		{
			*num=get_num_model_parameters();
			size_t sz=sizeof(hist)*(*num);
			*dst=(DREAL*) malloc(sz);
			ASSERT(dst);

			memcpy(*dst, hist, sz);
		}

	protected:
		DREAL* hist;
};
#endif
