/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef __STATE_MODEL_H__
#define __STATE_MODEL_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/structure/HMSVMLabels.h>
#include <shogun/structure/StateModelTypes.h>

namespace shogun
{

/** TODO DOC */
class CStateModel : public CSGObject
{
	public:
		/** default constructor */
		CStateModel();

		/** destructor */
		virtual ~CStateModel();

		/** @return number of states */
		int32_t get_num_states() const;

		/** @return number of transmission parameters to be learnt */
		int32_t get_num_transmission_params() const;

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
		virtual SGMatrix< float64_t > loss_matrix(CSequence* label_seq) = 0;

		/**
		 * computes the loss between two sequences of labels using the Hamming loss
		 * and the state loss matrix
		 *
		 * @param label_seq_lhs one label sequence
		 * @param label_seq_rhs other label sequence
		 *
		 * @return the Hamming loss
		 */
		virtual float64_t loss(CSequence* label_seq_lhs, CSequence* label_seq_rhs) = 0;

		/**
		 * arranges the emission parameterss of the weight vector into a vector
		 * adding zero elements for the states whose parameters are not learnt.
		 * This vector is suitable to iterate through when constructing the
		 * emission matrix used in Viterbi decoding
		 *
		 * @param w the weight vector
		 * @param num_feats number of features
		 * @param num_obs number of emission scores per feature and state
		 *
		 * @return a vector with the emission parameters
		 */
		virtual SGVector< float64_t > reshape_emission_params(SGVector< float64_t > w,
				int32_t num_feats, int32_t num_obs) = 0;

		/**
		 * arranges the tranmission parameterss of the weight vector into a matrix
		 * adding zero elements for the states whose parameters are not learnt.
		 * This matrix is suitable to iterate during Viterbi decoding
		 *
		 * @param w the weight vector
		 *
		 * @return a matrix with the transmission parameters
		 */
		virtual SGMatrix< float64_t > reshape_transmission_params(SGVector< float64_t > w) = 0;

		/** translates label sequence to state sequence
		 *
		 * @param label_seq label sequence
		 *
		 * @return state sequence
		 */
		virtual SGVector< int32_t > labels_to_states(CSequence* label_seq) const = 0;

		/** translates state sequence to label sequence 
		 *
		 * @param state_seq state sequence
		 *
		 * @return label sequence
		 */
		virtual CSequence* states_to_labels(SGVector< int32_t > state_seq) const = 0;

		/**
		 * reshapes the transition and emission weights into a vector (the joint
		 * feature vector so it will be possible to take the dot product with the
		 * weight vector)
		 *
		 * @param psi output vector
		 * @param transition_weights counts of the state transitions for a state
		 * sequence
		 * @param emission_weights counts of the emission scores for a state
		 * sequence and a feature vector
		 * @param num_feats number of features
		 * @param num_obs number of emission scores per feature and state
		 */
		virtual void weights_to_vector(SGVector< float64_t >& psi,
				SGMatrix< float64_t > transiton_weights,
				SGVector< float64_t > emission_weights,
				int32_t num_feats, int32_t num_obs) const = 0;

		/**
		 * specify monotonicity constraints for feature scoring functions. The
		 * elements of the vector returned can take one of three values:
		 *
		 * - +1 if monotonically increasing scoring function
		 * - -1 if monotonically decreasing scoring function
		 * -  0 if no constraint in the monotonicity
		 *
		 * By default, this method returns a vector of zeros (no monotonicity
		 * constraints). Re-implement this method in a child class to specify
		 * constraints.
		 *
		 * @param num_free_states number of states whose parameters are learnt
		 * @param num_feats number of features
		 *
		 * @return vector with monotonicity constraints of length num_feats times
		 * num_free_states
		 */
		virtual SGVector< int32_t > get_monotonicity(int32_t num_free_states,
				int32_t num_feats) const;

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "StateModel"; }

	private:
		/** internal initialization */
		void init();

	protected:
		/** the number of states */
		int32_t m_num_states;

		/** the number of transmission parameters */
		int32_t m_num_transmission_params;

		/** state loss matrix, loss for every pair of states */
		SGMatrix< float64_t > m_state_loss_mat;

}; /* class CStateModel */

} /* namespace shogun */

#endif /* __STATE_MODEL_H__ */
