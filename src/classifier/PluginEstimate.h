#ifndef _PLUGINESTIMATE_H___
#define _PLUGINESTIMATE_H___

#include "features/WordFeatures.h"
#include "features/Labels.h"
#include "distributions/hmm/LinearHMM.h"

class CPluginEstimate
{
	public:
		CPluginEstimate();
		~CPluginEstimate();

		bool train(CWordFeatures* features, CLabels* labels, REAL pos_pseudo, REAL neg_pseudo);
		REAL* test();

		void set_testfeatures(CWordFeatures* f) { test_features=f; }

		/// classify all test features
		CLabels* classify(CLabels* output=NULL);

		/// classify the test feature vector indexed by idx
		REAL classify_example(INT idx);

		inline REAL posterior_log_odds_obsolete(WORD* vector, INT len)
		{
			return pos_model->get_log_likelihood_example(vector, len) - neg_model->get_log_likelihood_example(vector, len);
		}

		inline REAL get_parameterwise_log_odds(WORD obs, INT position)
		{
			return pos_model->get_positional_log_parameter(obs, position) - neg_model->get_positional_log_parameter(obs, position);
		}

		inline REAL log_derivative_pos_obsolete(WORD obs, INT pos)
		{
			return pos_model->get_log_derivative_obsolete(obs, pos);
		}
		inline REAL log_derivative_neg_obsolete(WORD obs, INT pos)
		{
			return neg_model->get_log_derivative_obsolete(obs, pos);
		}

		inline bool get_model_params(REAL*& pos_params, REAL*& neg_params, INT &seq_length, INT &num_symbols)
		{
			if ((!pos_model) || (!neg_model))
			{
				CIO::message(M_ERROR, "no model available\n");
				return false;
			}

			pos_params = pos_model->get_log_hist();
			neg_params = neg_model->get_log_hist();

			seq_length = pos_model->get_sequence_length();
			num_symbols = pos_model->get_num_symbols();
			assert(pos_model->get_num_model_parameters() == neg_model->get_num_model_parameters());
			assert(pos_model->get_num_symbols() == neg_model->get_num_symbols());
			return true;
		}

		inline void set_model_params(const REAL* pos_params, const REAL* neg_params, INT seq_length, INT num_symbols)
		{
			if (pos_model)
				delete pos_model;

			pos_model = new CLinearHMM(seq_length, num_symbols);

			if (neg_model)
				delete neg_model;

			neg_model = new CLinearHMM(seq_length, num_symbols);

			assert(pos_model);
			assert(neg_model);

			assert(seq_length*num_symbols == pos_model->get_num_model_parameters());
			assert(pos_model->get_num_model_parameters() == neg_model->get_num_model_parameters());

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
		CWordFeatures* test_features;
};
#endif
