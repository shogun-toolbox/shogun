/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _PLUGINESTIMATE_H___
#define _PLUGINESTIMATE_H___

#include <shogun/lib/config.h>

#include <shogun/machine/Machine.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/distributions/LinearHMM.h>

namespace shogun
{
/** @brief class PluginEstimate
 *
 * The class PluginEstimate takes as input two probabilistic models (of type
 * CLinearHMM, even though general models are possible ) and classifies
 * examples according to the rule
 *
 * \f[
 * f({\bf x})= \log(\mbox{Pr}({\bf x}|\theta_+)) - \log(\mbox{Pr}({\bf x}|\theta_-))
 * \f]
 *
 * \sa CLinearHMM
 * \sa CDistribution
 * */
class CPluginEstimate: public CMachine
{
	public:

		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		/** default constructor
		 * @param pos_pseudo pseudo for positive model
		 * @param neg_pseudo pseudo for negative model
		 */
		CPluginEstimate(float64_t pos_pseudo=1e-10, float64_t neg_pseudo=1e-10);
		virtual ~CPluginEstimate();

		/** classify objects
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CBinaryLabels* apply_binary(CFeatures* data=NULL);

		/** set features
		 *
		 * @param feat features to set
		 */
		virtual void set_features(CStringFeatures<uint16_t>* feat)
		{
			SG_REF(feat);
			SG_UNREF(features);
			features=feat;
		}

		/** get features
		 *
		 * @return features
		 */
		virtual CStringFeatures<uint16_t>* get_features() { SG_REF(features); return features; }

		/// classify the test feature vector indexed by vec_idx
		float64_t apply_one(int32_t vec_idx);

		/** obsolete posterior log odds
		 *
		 * @param vector vector
		 * @param len len
		 * @return something floaty
		 */
		inline float64_t posterior_log_odds_obsolete(
			uint16_t* vector, int32_t len)
		{
			return pos_model->get_log_likelihood_example(vector, len) - neg_model->get_log_likelihood_example(vector, len);
		}

		/** get log odds parameter-wise
		 *
		 * @param obs observation
		 * @param position position
		 * @return log odd at position
		 */
		inline float64_t get_parameterwise_log_odds(
			uint16_t obs, int32_t position)
		{
			return pos_model->get_positional_log_parameter(obs, position) - neg_model->get_positional_log_parameter(obs, position);
		}

		/** get obsolete positive log derivative
		 *
		 * @param obs observation
		 * @param pos position
		 * @return positive log derivative
		 */
		inline float64_t log_derivative_pos_obsolete(uint16_t obs, int32_t pos)
		{
			return pos_model->get_log_derivative_obsolete(obs, pos);
		}

		/** get obsolete negative log derivative
		 *
		 * @param obs observation
		 * @param pos position
		 * @return negative log derivative
		 */
		inline float64_t log_derivative_neg_obsolete(uint16_t obs, int32_t pos)
		{
			return neg_model->get_log_derivative_obsolete(obs, pos);
		}

		/** get model parameters
		 *
		 * @param pos_params parameters of positive model
		 * @param neg_params parameters of negative model
		 * @param seq_length sequence length
		 * @param num_symbols numbe of symbols
		 * @return if operation was successful
		 */
		inline bool get_model_params(
			float64_t*& pos_params, float64_t*& neg_params,
			int32_t &seq_length, int32_t &num_symbols)
		{
			if ((!pos_model) || (!neg_model))
			{
				SG_ERROR("no model available\n")
				return false;
			}

			SGVector<float64_t> log_pos_trans = pos_model->get_log_transition_probs();
			pos_params = log_pos_trans.vector;
			SGVector<float64_t> log_neg_trans = neg_model->get_log_transition_probs();
			neg_params = log_neg_trans.vector;

			seq_length = pos_model->get_sequence_length();
			num_symbols = pos_model->get_num_symbols();
			ASSERT(pos_model->get_num_model_parameters()==neg_model->get_num_model_parameters())
			ASSERT(pos_model->get_num_symbols()==neg_model->get_num_symbols())
			return true;
		}

		/** set model parameters
		 * @param pos_params parameters of positive model
		 * @param neg_params parameters of negative model
		 * @param seq_length sequence length
		 * @param num_symbols numbe of symbols
		 */
		inline void set_model_params(
			float64_t* pos_params, float64_t* neg_params,
			int32_t seq_length, int32_t num_symbols)
		{
			int32_t num_params;

			SG_UNREF(pos_model);
			pos_model=new CLinearHMM(seq_length, num_symbols);
			SG_REF(pos_model);


			SG_UNREF(neg_model);
			neg_model=new CLinearHMM(seq_length, num_symbols);
			SG_REF(neg_model);

			num_params=pos_model->get_num_model_parameters();
			ASSERT(seq_length*num_symbols==num_params)
			ASSERT(num_params==neg_model->get_num_model_parameters())

			pos_model->set_log_transition_probs(SGVector<float64_t>(pos_params, num_params));
			neg_model->set_log_transition_probs(SGVector<float64_t>(neg_params, num_params));
		}

		/** get number of parameters
		 *
		 * @return number of parameters
		 */
		inline int32_t get_num_params()
		{
			return pos_model->get_num_model_parameters()+neg_model->get_num_model_parameters();
		}

		/** check models
		 *
		 * @return if one of the two models is invalid
		 */
		inline bool check_models()
		{
			return ( (pos_model!=NULL) && (neg_model!=NULL) );
		}

		/** @return object name */
		virtual const char* get_name() const { return "PluginEstimate"; }

	protected:
		/** train plugin estimate classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

	protected:
		/** pseudo count for positive class */
		float64_t m_pos_pseudo;
		/** pseudo count for negative class */
		float64_t m_neg_pseudo;

		/** positive model */
		CLinearHMM* pos_model;
		/** negative model */
		CLinearHMM* neg_model;

		/** features */
		CStringFeatures<uint16_t>* features;
};
}
#endif
