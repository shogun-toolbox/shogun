/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LINEARHMM_H__
#define _LINEARHMM_H__

#include "features/StringFeatures.h"
#include "features/Labels.h"
#include "distributions/Distribution.h"

class CLinearHMM : public CDistribution
{
	public:
		CLinearHMM(CStringFeatures<WORD>* f);
		CLinearHMM(INT p_num_features, INT p_num_symbols);
		~CLinearHMM();

		bool train();
		bool train(const INT* indizes, INT num_indizes, DREAL pseudo_count);

		DREAL get_log_likelihood_example(WORD* vector, INT len);
		DREAL get_likelihood_example(WORD* vector, INT len);

		virtual DREAL get_log_likelihood_example(INT num_example);

		virtual DREAL get_log_derivative(INT num_param, INT num_example);

		virtual inline DREAL get_log_derivative_obsolete(WORD obs, INT pos)
		{
			return 1.0/transition_probs[pos*num_symbols+obs];
		}

		virtual inline DREAL get_derivative_obsolete(WORD* vector, INT len, INT pos)
		{
			ASSERT(pos<len);
			return get_likelihood_example(vector, len)/transition_probs[pos*num_symbols+vector[pos]];
		}

		virtual inline INT get_sequence_length() { return sequence_length; }

		virtual inline INT get_num_symbols() { return num_symbols; }

		virtual inline INT get_num_model_parameters() { return num_params; }

		virtual inline DREAL get_positional_log_parameter(WORD obs, INT position)
		{
			return log_transition_probs[position*num_symbols+obs];
		}

		virtual inline DREAL get_log_model_parameter(INT num_param)
		{
			ASSERT(log_transition_probs);
			ASSERT(num_param<num_params);

			return log_transition_probs[num_param];
		}

		virtual void get_log_transition_probs(DREAL** dst, INT* num);
		virtual bool set_log_transition_probs(const DREAL* src, INT num);

		virtual void get_transition_probs(DREAL** dst, INT* num);
		virtual bool set_transition_probs(const DREAL* src, INT num);

	protected:
		INT sequence_length;
		INT num_symbols;
		INT num_params;
		DREAL* transition_probs;
		DREAL* log_transition_probs;
};
#endif
