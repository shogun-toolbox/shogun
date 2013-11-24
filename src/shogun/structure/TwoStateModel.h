/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef __TWO_STATE_MODEL_H__
#define __TWO_STATE_MODEL_H__

#include <shogun/structure/StateModel.h>
#include <shogun/structure/HMSVMModel.h>

namespace shogun
{

/**
 * @brief class CTwoStateModel class for the internal two-state representation
 * used in the CHMSVMModel.
 */
class CTwoStateModel : public CStateModel
{
	public:
		/** default constructor */
		CTwoStateModel();

		/** destructor */
		virtual ~CTwoStateModel();

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
		virtual SGMatrix< float64_t > loss_matrix(Sequence* label_seq);

		/**
		 * computes the loss between two sequences of labels using the Hamming loss
		 * and the state loss matrix
		 *
		 * @param label_seq_lhs one label sequence
		 * @param label_seq_rhs other label sequence
		 *
		 * @return the Hamming loss
		 */
		virtual float64_t loss(Sequence* label_seq_lhs, Sequence* label_seq_rhs);

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
		virtual void reshape_emission_params(SGVector< float64_t >& emission_weights,
				SGVector< float64_t > w, int32_t num_feats, int32_t num_obs);

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
		virtual void reshape_emission_params(CDynamicObjectArray* plif_matrix,
			SGVector< float64_t > w, int32_t num_feats, int32_t num_plif_nodes);

		/**
		 * arranges the transmission parameters of the weight vector into a matrix
		 * adding zero elements for the states whose parameters are not learnt.
		 * This matrix is suitable to iterate during Viterbi decoding
		 *
		 * @param transmission_weights transmission parameters outputted
		 * @param w the weight vector
		 */
		virtual void reshape_transmission_params(
				SGMatrix< float64_t >& transmission_weights,
				SGVector< float64_t > w);

		/** translates label sequence to state sequence
		 *
		 * @param label_seq label sequence
		 *
		 * @return state sequence
		 */
		virtual SGVector< int32_t > labels_to_states(Sequence* label_seq) const;

		/** translates state sequence to label sequence
		 *
		 * @param state_seq state sequence
		 *
		 * @return label sequence
		 */
		virtual Sequence* states_to_labels(SGVector< int32_t > state_seq) const;

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
		virtual void weights_to_vector(SGVector< float64_t >& psi,
				SGMatrix< float64_t > transmission_weights,
				SGVector< float64_t > emission_weights,
				int32_t num_feats, int32_t num_obs) const;

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
		virtual SGVector< float64_t > weights_to_vector(SGMatrix< float64_t > transmission_weights,
				SGVector< float64_t > emission_weights, int32_t num_feats, int32_t num_obs) const;

		/**
		 * specify monotonicity constraints for feature scoring functions. The
		 * elements of the vector returned can take one of three values:
		 *
		 * see CStateModel::get_monotonicity
		 *
		 * @param num_free_states number of states whose parameters are learnt
		 * @param num_feats number of features
		 *
		 * @return vector with monotonicity constraints of length num_feats times
		 * num_learnt_states
		 */
		virtual SGVector< int32_t > get_monotonicity(int32_t num_free_states,
				int32_t num_feats) const;

		/**
		 * generates simulated data. The features are generated from the label
		 * sequence swapping some of the labels and adding noise
		 *
		 * @param num_exm number of sample pairs (sequence of features, sequence of labels) to generate
		 * @param exm_len length of each sample sequence
		 * @param num_features features dimension
		 * @param num_noise_features number of features to be pure noise
		 *
		 * @return a model that contains the data simulated
		 */
		static CHMSVMModel* simulate_data(int32_t num_exm, int32_t exm_len, int32_t num_features, int32_t num_noise_features);

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "TwoStateModel"; }
};

} /* namespace shogun */

#endif /* __TWO_STATE_MODEL_H__ */
