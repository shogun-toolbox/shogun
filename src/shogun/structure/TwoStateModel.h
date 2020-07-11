/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Yuyu Zhang
 */

#ifndef __TWO_STATE_MODEL_H__
#define __TWO_STATE_MODEL_H__

#include <shogun/lib/config.h>

#include <shogun/structure/StateModel.h>
#include <shogun/structure/HMSVMModel.h>
#include <shogun/mathematics/RandomMixin.h>
#include <shogun/mathematics/NormalDistribution.h>

namespace shogun
{

/**
 * @brief class TwoStateModel class for the internal two-state representation
 * used in the CHMSVMModel.
 */
class TwoStateModel : public StateModel
{
	public:
		/** default constructor */
		TwoStateModel();

		/** destructor */
		~TwoStateModel() override;

		/**
		 * computes a loss matrix with m_num_states rows and number of columns
		 * equal to the length of label_seq. This matrix can be added directly
		 * to the emission matrix used in Viterbi decoding during training to
		 * form the loss-augmented emission matrix
		 *
		 * @param label_seq label sequence (normally the true label sequence)
		 *
		 * @return the loss matrix
		 */
		SGMatrix< float64_t > loss_matrix(std::shared_ptr<Sequence> label_seq) override;

		/**
		 * computes the loss between two sequences of labels using the Hamming loss
		 * and the state loss matrix
		 *
		 * @param label_seq_lhs one label sequence
		 * @param label_seq_rhs other label sequence
		 *
		 * @return the Hamming loss
		 */
		float64_t loss(std::shared_ptr<Sequence> label_seq_lhs, std::shared_ptr<Sequence> label_seq_rhs) override;

		/**
		 * arranges the emission parameterss of the weight vector into a vector
		 * adding zero elements for the states whose parameters are not learnt.
		 * This vector is suitable to iterate through when constructing the
		 * emission matrix used in Viterbi decoding
		 *
		 * @param emission_weights emission parameters outputted
		 * @param w the weight vector
		 * @param num_feats number of features
		 * @param num_obs number of emission scores per feature and state
		 */
		void reshape_emission_params(SGVector< float64_t >& emission_weights,
				SGVector< float64_t > w, int32_t num_feats, int32_t num_obs) override;

		/**
		 * arranges the emission parameters of the weight vector into a matrix
		 * of PLiFs adding zero elements for the states whose parameters are not
		 * learnt.
		 *
		 * @param plif_matrix matrix of PLiFs outputted
		 * @param w the weight vector
		 * @param num_feats number of features
		 * @param num_plif_nodes number of nodes in the PLiFs
		 */
		void reshape_emission_params(const std::vector<std::shared_ptr<Plif>>& plif_matrix,
			SGVector< float64_t > w, int32_t num_feats, int32_t num_plif_nodes) override;

		/**
		 * arranges the transmission parameters of the weight vector into a matrix
		 * adding zero elements for the states whose parameters are not learnt.
		 * This matrix is suitable to iterate during Viterbi decoding
		 *
		 * @param transmission_weights transmission parameters outputted
		 * @param w the weight vector
		 */
		void reshape_transmission_params(
				SGMatrix< float64_t >& transmission_weights,
				SGVector< float64_t > w) override;

		/** translates label sequence to state sequence
		 *
		 * @param label_seq label sequence
		 *
		 * @return state sequence
		 */
		SGVector< int32_t > labels_to_states(std::shared_ptr<Sequence> label_seq) const override;

		/** translates state sequence to label sequence
		 *
		 * @param state_seq state sequence
		 *
		 * @return label sequence
		 */
		std::shared_ptr<Sequence> states_to_labels(SGVector< int32_t > state_seq) const override;

		/**
		 * reshapes the transition and emission weights into a vector (the joint
		 * feature vector so it will be possible to take the dot product with the
		 * weight vector). Version with the joint feature vector as parameter by
		 * reference
		 *
		 * @param psi output vector
		 * @param transmission_weights counts of the state transitions for a state sequence
		 * @param emission_weights counts of the emission scores for a state sequence and a feature vector
		 * @param num_feats number of features
		 * @param num_obs number of emission scores per feature and state
		 */
		void weights_to_vector(SGVector< float64_t >& psi,
				SGMatrix< float64_t > transmission_weights,
				SGVector< float64_t > emission_weights,
				int32_t num_feats, int32_t num_obs) const override;

		/**
		 * reshapes the transition and emission weights into a vector (the joint
		 * feature vector so it will be possible to take the dot product with the
		 * weight vector). Version returning the joint feature vector
		 *
		 * @param transmission_weights counts of the state transitions for a state sequence
		 * @param emission_weights counts of the emission scores for a state sequence and a feature vector
		 * @param num_feats number of features
		 * @param num_obs number of emission scores per feature and state
		 *
		 * @return psi output vector
		 */
		SGVector< float64_t > weights_to_vector(SGMatrix< float64_t > transmission_weights,
				SGVector< float64_t > emission_weights, int32_t num_feats, int32_t num_obs) const override;

		/**
		 * specify monotonicity constraints for feature scoring functions. The
		 * elements of the vector returned can take one of three values:
		 *
		 * see StateModel::get_monotonicity
		 *
		 * @param num_free_states number of states whose parameters are learnt
		 * @param num_feats number of features
		 *
		 * @return vector with monotonicity constraints of length num_feats times
		 * num_learnt_states
		 */
		SGVector< int32_t > get_monotonicity(int32_t num_free_states,
				int32_t num_feats) const override;

		/**
		 * generates simulated data. The features are generated from the label
		 * sequence swapping some of the labels and adding noise
		 *
		 * @param num_exm number of sample pairs (sequence of features, sequence of labels) to generate
		 * @param exm_len length of each sample sequence
		 * @param num_features features dimension
		 * @param num_noise_features number of features to be pure noise
		 * @param prng random number generator
		 *
		 * @return a model that contains the data simulated
		 */
		template <typename PRNG>
		static std::shared_ptr<HMSVMModel> simulate_data(
			int32_t num_exm, int32_t exm_len, int32_t num_features,
			int32_t num_noise_features, PRNG& prng);

		/**
		 * generates simulated data. The features are generated from the label
		 * sequence swapping some of the labels and adding noise
		 *
		 * @param num_exm number of sample pairs (sequence of features, sequence of labels) to generate
		 * @param exm_len length of each sample sequence
		 * @param num_features features dimension
		 * @param num_noise_features number of features to be pure noise
		 * @param seed seed for the random number generator
		 *
		 * @return a model that contains the data simulated
		 */
		static std::shared_ptr<HMSVMModel> simulate_data(
			int32_t num_exm, int32_t exm_len, int32_t num_features,
			int32_t num_noise_features, int32_t seed=-1);

		/** @return name of SGSerializable */
		const char* get_name() const override { return "TwoStateModel"; }
};

} /* namespace shogun */

#endif /* __TWO_STATE_MODEL_H__ */
