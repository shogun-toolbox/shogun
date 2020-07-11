/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Yuyu Zhang
 */

#ifndef _LINEARHMM_H__
#define _LINEARHMM_H__

#include <shogun/lib/config.h>

#include <shogun/features/StringFeatures.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/distributions/Distribution.h>

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
class LinearHMM : public Distribution
{
	public:
		/** default constructor  */
		LinearHMM();

		/** constructor
		 *
		 * @param f features to use
		 */
		LinearHMM(const std::shared_ptr<StringFeatures<uint16_t>>& f);

		/** constructor
		 *
		 * @param p_num_features number of features
		 * @param p_num_symbols number of symbols in features
		 */
		LinearHMM(int32_t p_num_features, int32_t p_num_symbols);

		~LinearHMM() override;

		/** estimate LinearHMM distribution
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		bool train(std::shared_ptr<Features> data=NULL) override;

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
		float64_t get_likelihood_example(int32_t num_example) override;

		/** get logarithm of one example's likelihood
		 *
		 * @param num_example which example
		 * @return logarithm of example's likelihood
		 */
		float64_t get_log_likelihood_example(int32_t num_example) override;

		/** get logarithm of one example's derivative's likelihood
		 *
		 * @param num_param which example's param
		 * @param num_example which example
		 * @return logarithm of example's derivative
		 */
		float64_t get_log_derivative(
			int32_t num_param, int32_t num_example) override;

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
		int32_t get_num_model_parameters() override { return num_params; }

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
		float64_t get_log_model_parameter(int32_t num_param) override
		{
			ASSERT(log_transition_probs.size() == num_params)
			ASSERT(num_param<num_params)

			return log_transition_probs[num_param];
		}

		/** get logarithm of all transition probs
		 *
		 * @return logarithm of transition probs vector
		 */
		virtual SGMatrix<float64_t> get_log_transition_probs();

		/** set logarithm of all transition probs
		 *
		 * @param probs new logarithm transition probs
		 * @return if setting was successful
		 */
		virtual bool set_log_transition_probs(const SGMatrix<float64_t>& probs);

		/** get all transition probs
		 *
		 * @return vector of transition probs
		 */
		virtual SGMatrix<float64_t> get_transition_probs();

		/** set all transition probs
		 *
		 * @param probs new transition probs
		 * @return if setting was successful
		 */
		virtual bool set_transition_probs(const SGMatrix<float64_t>& probs);

		/** @return object name */
		const char* get_name() const override { return "LinearHMM"; }

		/** set feature vectors
		 *
		 * @param f new feature vectors
		 */
		void set_features(std::shared_ptr<Features> f) override;

	protected:
		void load_serializable_post() override;

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
		SGMatrix<float64_t> transition_probs;
		/** logarithm of transition probs */
		SGMatrix<float64_t> log_transition_probs;
};
}
#endif
