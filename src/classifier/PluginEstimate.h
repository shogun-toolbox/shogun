/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _PLUGINESTIMATE_H___
#define _PLUGINESTIMATE_H___

#include "base/SGObject.h"
#include "features/StringFeatures.h"
#include "features/Labels.h"
#include "distributions/hmm/LinearHMM.h"

class CPluginEstimate: public CSGObject
{
	public:
		CPluginEstimate();
		~CPluginEstimate();

		bool train(CStringFeatures<WORD>* features, CLabels* labels, DREAL pos_pseudo, DREAL neg_pseudo);
		DREAL* test();

		void set_testfeatures(CStringFeatures<WORD>* f) { test_features=f; }

		/// classify all test features
		CLabels* classify(CLabels* output=NULL);

		/// classify the test feature vector indexed by idx
		DREAL classify_example(INT idx);

		inline DREAL posterior_log_odds_obsolete(WORD* vector, INT len)
		{
			return pos_model->get_log_likelihood_example(vector, len) - neg_model->get_log_likelihood_example(vector, len);
		}

		inline DREAL get_parameterwise_log_odds(WORD obs, INT position)
		{
			return pos_model->get_positional_log_parameter(obs, position) - neg_model->get_positional_log_parameter(obs, position);
		}

		inline DREAL log_derivative_pos_obsolete(WORD obs, INT pos)
		{
			return pos_model->get_log_derivative_obsolete(obs, pos);
		}
		inline DREAL log_derivative_neg_obsolete(WORD obs, INT pos)
		{
			return neg_model->get_log_derivative_obsolete(obs, pos);
		}

		inline bool get_model_params(DREAL*& pos_params, DREAL*& neg_params, INT &seq_length, INT &num_symbols)
		{
			if ((!pos_model) || (!neg_model))
			{
				SG_ERROR( "no model available\n");
				return false;
			}

			pos_params = pos_model->get_log_hist();
			neg_params = neg_model->get_log_hist();

			seq_length = pos_model->get_sequence_length();
			num_symbols = pos_model->get_num_symbols();
			ASSERT(pos_model->get_num_model_parameters() == neg_model->get_num_model_parameters());
			ASSERT(pos_model->get_num_symbols() == neg_model->get_num_symbols());
			return true;
		}

		inline void set_model_params(const DREAL* pos_params, const DREAL* neg_params, INT seq_length, INT num_symbols)
		{
			if (pos_model)
				delete pos_model;

			pos_model = new CLinearHMM(seq_length, num_symbols);

			if (neg_model)
				delete neg_model;

			neg_model = new CLinearHMM(seq_length, num_symbols);

			ASSERT(pos_model);
			ASSERT(neg_model);

			ASSERT(seq_length*num_symbols == pos_model->get_num_model_parameters());
			ASSERT(pos_model->get_num_model_parameters() == neg_model->get_num_model_parameters());

			pos_model->set_log_hist(pos_params);
			neg_model->set_log_hist(neg_params);
		}

		inline INT get_num_params()
		{
			return pos_model->get_num_model_parameters()+neg_model->get_num_model_parameters();
		}
		
		inline bool check_models()
		{
			return ( (pos_model!=NULL) && (neg_model!=NULL) );
		}

	protected:
		CLinearHMM* pos_model;
		CLinearHMM* neg_model;
		CStringFeatures<WORD>* test_features;
};
#endif
