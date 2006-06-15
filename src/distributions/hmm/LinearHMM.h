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

#ifndef _LINEARHMM_H__
#define _LINEARHMM_H__

#include "features/WordFeatures.h"
#include "features/Labels.h"
#include "distributions/Distribution.h"

class CLinearHMM : private CDistribution
{
	public:
		CLinearHMM(CWordFeatures* f);
		CLinearHMM(INT p_num_features, INT p_num_symbols);
		~CLinearHMM();

		bool train();
		bool train(const INT* indizes, INT num_indizes, DREAL pseudo_count);
		bool marginalized_train(const INT* indizes, INT num_indizes, DREAL pseudo_count, INT order);

		DREAL get_log_likelihood_example(WORD* vector, INT len);
		DREAL get_likelihood_example(WORD* vector, INT len);

		virtual DREAL get_log_likelihood_example(INT num_example);

		virtual DREAL get_log_derivative(INT param_num, INT num_example);

		virtual inline DREAL get_log_derivative_obsolete(WORD obs, INT pos)
		{
			return 1.0/hist[pos*num_symbols+obs];
		}

		virtual inline DREAL get_derivative_obsolete(WORD* vector, INT len, INT pos)
		{
			ASSERT(pos<len);
			return get_likelihood_example(vector, len)/hist[pos*num_symbols+vector[pos]];
		}

		inline INT get_sequence_length() { return sequence_length; }

		inline INT get_num_symbols() { return num_symbols; }

		inline INT get_num_model_parameters() { return num_params; }

		inline DREAL get_positional_log_parameter(WORD obs, INT position)
		{
			return log_hist[position*num_symbols+obs];
		}

		inline DREAL get_log_model_parameter(INT param_num)
		{
			ASSERT(log_hist);
			ASSERT(param_num<num_params);

			return log_hist[param_num];
		}

		inline DREAL* get_log_hist() { return log_hist; }
		inline DREAL* get_hist() { return hist; }

		void set_log_hist(const DREAL* new_log_hist);
		void set_hist(const DREAL* new_hist);

	protected:
		INT sequence_length;
		INT num_symbols;
		INT num_params;
		DREAL* hist;
		DREAL* log_hist;
		CWordFeatures* features;
};
#endif
