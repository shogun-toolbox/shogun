/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LINEARHMM_H__
#define _LINEARHMM_H__

#include <features/StringFeatures.h>
#include <labels/Labels.h>
#include <distributions/Distribution.h>

namespace shogun
{
/** @brief The class LinearHMM is for learning Higher Order Markov chains.
 *
 * While learning the parameters \f${\bf \theta}\f$ in
 *
 * \f{eqnarray*}
 * P({\bf x}|{\bf \theta}^\pm)&=&P(x_1, \ldots, x_N|{\bf \theta}^\pm)\\
 * &=&P(x_1,\ldots,x_{d}|{\bf \theta}^\pm)\prod_{i=d+1}^N
 * P(x_i|x_{i-1},\ldots,x_{i-d},{\bf \theta}^\pm)
 * \f}
 *
 * are determined.
 *
 * A more detailed description can be found in
 *
 * Durbin et.al, Biological Sequence Analysis -Probabilistic Models of Proteins
 * and Nucleic Acids, 1998
 *
 * */
class CLinearHMM : public CDistribution
{
	public:
		/** default constructor  */
		CLinearHMM();

		/** constructor
		 *
		 * @param f features to use
		 */
		CLinearHMM(CStringFeatures<uint16_t>* f);

		/** constructor
		 *
		 * @param p_num_features number of features
		 * @param p_num_symbols number of symbols in features
		 */
		CLinearHMM(int32_t p_num_features, int32_t p_num_symbols);

		virtual ~CLinearHMM();

		/** estimate LinearHMM distribution
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/** alternative train distribution
		 *
		 * @param indizes indices
		 * @param num_indizes number of indices
		 * @param pseudo_count pseudo count
		 * @return if training was successful
		 */
		bool train(
			const int32_t* indizes, int32_t num_indizes,
			float64_t pseudo_count);

		/** get logarithm of one example's likelihood
		 *
		 * @param vector the example
		 * @param len length of vector
		 * @return logarithm of likelihood
		 */
		float64_t get_log_likelihood_example(uint16_t* vector, int32_t len);

		/** get one example's likelihood
		 *
		 * @param vector the example
		 * @param len length of vector
		 * @return likelihood
		 */
		float64_t get_likelihood_example(uint16_t* vector, int32_t len);

		/** compute likelihood for example
		 *
		 * @param num_example which example
		 * @return likelihood for example
		 */
		float64_t get_likelihood_example(int32_t num_example);

		/** get logarithm of one example's likelihood
		 *
		 * @param num_example which example
		 * @return logarithm of example's likelihood
		 */
		virtual float64_t get_log_likelihood_example(int32_t num_example);

		/** get logarithm of one example's derivative's likelihood
		 *
		 * @param num_param which example's param
		 * @param num_example which example
		 * @return logarithm of example's derivative
		 */
		virtual float64_t get_log_derivative(
			int32_t num_param, int32_t num_example);

		/** obsolete get logarithm of one example's derivative's
		 *  likelihood
		 *
		 * @param obs observation
		 * @param pos position
		 */
		virtual float64_t get_log_derivative_obsolete(
			uint16_t obs, int32_t pos)
		{
			return 1.0/transition_probs[pos*num_symbols+obs];
		}

		/** obsolete get one example's derivative
		 *
		 * @param vector vector
		 * @param len length
		 * @param pos position
		 */
		virtual float64_t get_derivative_obsolete(
			uint16_t* vector, int32_t len, int32_t pos)
		{
			ASSERT(pos<len)
			return get_likelihood_example(vector, len)/transition_probs[pos*num_symbols+vector[pos]];
		}

		/** get sequence length of each example
		 *
		 * @return sequence length of each example
		 */
		virtual int32_t get_sequence_length() { return sequence_length; }

		/** get number of symbols in examples
		 *
		 * @return number of symbols in examples
		 */
		virtual int32_t get_num_symbols() { return num_symbols; }

		/** get number of model parameters
		 *
		 * @return number of model parameters
		 */
		virtual int32_t get_num_model_parameters() { return num_params; }

		/** get positional log parameter
		 *
		 * @param obs observation
		 * @param position position
		 * @return positional log parameter
		 */
		virtual float64_t get_positional_log_parameter(
			uint16_t obs, int32_t position)
		{
			return log_transition_probs[position*num_symbols+obs];
		}

		/** get logarithm of given model parameter
		 *
		 * @param num_param which param
		 * @result logarithm of given model parameter
		 */
		virtual float64_t get_log_model_parameter(int32_t num_param)
		{
			ASSERT(log_transition_probs)
			ASSERT(num_param<num_params)

			return log_transition_probs[num_param];
		}

		/** get logarithm of all transition probs
		 *
		 * @return logarithm of transition probs vector
		 */
		virtual SGVector<float64_t> get_log_transition_probs();

		/** set logarithm of all transition probs
		 *
		 * @param probs new logarithm transition probs
		 * @return if setting was successful
		 */
		virtual bool set_log_transition_probs(const SGVector<float64_t> probs);

		/** get all transition probs
		 *
		 * @return vector of transition probs
		 */
		virtual SGVector<float64_t> get_transition_probs();

		/** set all transition probs
		 *
		 * @param probs new transition probs
		 * @return if setting was successful
		 */
		virtual bool set_transition_probs(const SGVector<float64_t> probs);

		/** @return object name */
		virtual const char* get_name() const { return "LinearHMM"; }

	protected:
		virtual void load_serializable_post() throw (ShogunException);

	private:
		void init();

	protected:
		/** examples' sequence length */
		int32_t sequence_length;
		/** number of symbols in examples */
		int32_t num_symbols;
		/** number of parameters */
		int32_t num_params;
		/** transition probs */
		float64_t* transition_probs;
		/** logarithm of transition probs */
		float64_t* log_transition_probs;
};
}
#endif
