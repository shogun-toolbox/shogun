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
		bool train(const INT* indizes, INT num_indizes, REAL pseudo_count);
		bool marginalized_train(const INT* indizes, INT num_indizes, REAL pseudo_count, INT order);

		REAL get_log_likelihood_example(WORD* vector, INT len);
		REAL get_likelihood_example(WORD* vector, INT len);

		virtual REAL get_log_likelihood_example(INT num_example);

		virtual REAL get_log_derivative(INT param_num, INT num_example);

		virtual inline REAL get_log_derivative_obsolete(WORD obs, INT pos)
		{
			return 1.0/hist[pos*num_symbols+obs];
		}

		virtual inline REAL get_derivative_obsolete(WORD* vector, INT len, INT pos)
		{
			assert(pos<len);
			return get_likelihood_example(vector, len)/hist[pos*num_symbols+vector[pos]];
		}

		inline INT get_sequence_length() { return sequence_length; }

		inline INT get_num_symbols() { return num_symbols; }

		inline INT get_num_model_parameters() { return num_params; }

		inline REAL get_positional_log_parameter(WORD obs, INT position)
		{
			return log_hist[position*num_symbols+obs];
		}

		inline REAL get_log_model_parameter(INT param_num)
		{
			assert(log_hist);
			assert(param_num<num_params);

			return log_hist[param_num];
		}

		inline REAL* get_log_hist() { return log_hist; }
		inline REAL* get_hist() { return hist; }

		void set_log_hist(REAL* new_log_hist);
		void set_hist(REAL* new_hist);

	protected:
		INT sequence_length;
		INT num_symbols;
		INT num_params;
		REAL* hist;
		REAL* log_hist;
		CWordFeatures* features;
};
#endif
