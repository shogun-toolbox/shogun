/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/structure/TwoStateModel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/MatrixFeatures.h>
#include <shogun/structure/Plif.h>

using namespace shogun;

CTwoStateModel::CTwoStateModel() : CStateModel()
{
	// The number of states in this state model is equal to four.
	// Although parameters are learnt only for two of them, other
	// two states (start and stop) are used
	m_num_states = 4;
	m_num_transmission_params = 4;

	m_state_loss_mat = SGMatrix< float64_t >(m_num_states, m_num_states);
	m_state_loss_mat.zero();
	for (index_t i = 0; i < m_num_states - 1; ++i)
	{
		m_state_loss_mat(m_num_states-1, i) = 1;
		m_state_loss_mat(i, m_num_states-1) = 1;
	}

	// Initialize the start and stop states
	m_p = SGVector< float64_t >(m_num_states);
	m_q = SGVector< float64_t >(m_num_states);
	m_p.set_const(-CMath::INFTY);
	m_q.set_const(-CMath::INFTY);
	m_p[0] = 0; // start state
	m_q[1] = 0; // stop  state
}

CTwoStateModel::~CTwoStateModel()
{
}

SGMatrix< float64_t > CTwoStateModel::loss_matrix(CSequence* label_seq)
{
	SGVector<index_t> state_seq = labels_to_states(label_seq);
	SGMatrix< float64_t > loss_mat(m_num_states, state_seq.vlen);

	for (index_t i = 0; i < loss_mat.num_cols; ++i)
	{
		for (index_t s = 0; s < loss_mat.num_rows; ++s)
			loss_mat(s,i) = m_state_loss_mat(s, state_seq[i]);
	}

	return loss_mat;
}

float64_t CTwoStateModel::loss(CSequence* label_seq_lhs, CSequence* label_seq_rhs)
{
	SGVector<index_t> state_seq_lhs = labels_to_states(label_seq_lhs);
	SGVector<index_t> state_seq_rhs = labels_to_states(label_seq_rhs);

	ASSERT(state_seq_lhs.vlen == state_seq_rhs.vlen)

	float64_t ret = 0.0;
	for (index_t i = 0; i < state_seq_lhs.vlen; ++i)
		ret += m_state_loss_mat(state_seq_lhs[i], state_seq_rhs[i]);

	return ret;
}

SGVector<index_t> CTwoStateModel::labels_to_states(CSequence* label_seq) const
{
	// 0 -> start state
	// 1 -> stop state
	// 2 -> negative state (label == 0)
	// 3 -> positive state (label == 1)

	SGVector<index_t> seq_data = label_seq->get_data();
	SGVector<index_t> state_seq(seq_data.size());
	for (index_t i = 1; i < state_seq.vlen - 1; ++i)
	{
		//FIXME make independent of values 0-1 in labels
		state_seq[i] = seq_data[i] + 2;
	}

	// The first element is always start state
	state_seq[0] = 0;
	// The last element is always stop state
	state_seq[state_seq.vlen-1] = 1;

	return state_seq;
}

CSequence* CTwoStateModel::states_to_labels(SGVector<index_t> state_seq) const
{
	SGVector<index_t> label_seq(state_seq.vlen);

	//FIXME make independent of values 0-1 in labels
	// Legend for state indices:
	// 0 -> start state => label 0
	// 1 -> stop state => label 0
	// 2 -> negative state (label == 0) => label 0
	// 3 -> positive state (label == 1) => label 1
	label_seq.zero();
	for (index_t i = 0; i < state_seq.vlen; ++i)
	{
		if ( state_seq[i] == 3 )
			label_seq[i] = 1;
	}

	CSequence* ret = new CSequence(label_seq);
	SG_REF(ret);
	return ret;
}

void CTwoStateModel::reshape_emission_params(
    SGVector<float64_t>& emission_weights, SGVector<float64_t> w,
    index_t num_feats, index_t num_obs)
{
	emission_weights.zero();

	// Legend for state indices:
	// 0 -> start state
	// 1 -> stop state
	// 2 -> negative state (label == 0)
	// 3 -> positive state (label == 1)
	//
	// start and stop states have no emission scores

	index_t em_idx, w_idx = m_num_transmission_params;
	for (index_t s = 2; s < m_num_states; ++s)
	{
		for (index_t f = 0; f < num_feats; ++f)
		{
			for (index_t o = 0; o < num_obs; ++o)
			{
				em_idx = s*num_feats*num_obs + f*num_obs + o;
				emission_weights[em_idx] = w[w_idx++];
			}
		}
	}
}

void CTwoStateModel::reshape_emission_params(
    CDynamicObjectArray* plif_matrix, SGVector<float64_t> w, index_t num_feats,
    index_t num_plif_nodes)
{
	CPlif* plif;
	index_t p_idx, w_idx = m_num_transmission_params;
	for (index_t s = 2; s < m_num_states; ++s)
	{
		for (index_t f = 0; f < num_feats; ++f)
		{
			SGVector< float64_t > penalties(num_plif_nodes);
			p_idx = 0;

			for (index_t i = 0; i < num_plif_nodes; ++i)
				penalties[p_idx++] = w[w_idx++];

			plif = (CPlif*) plif_matrix->get_element(m_num_states*f + s);
			plif->set_plif_penalty(penalties);
			SG_UNREF(plif);
		}
	}
}

void CTwoStateModel::reshape_transmission_params(
		SGMatrix< float64_t >& transmission_weights, SGVector< float64_t > w)
{
	transmission_weights.set_const(-CMath::INFTY);

	// Legend for state indices:
	// 0 -> start state
	// 1 -> stop state
	// 2 -> negative state (label == 0)
	// 3 -> positive state (label == 1)

	// From start
	transmission_weights(0,2) = 0;    // to negative
	transmission_weights(0,3) = 0;    // to positive
	// From negative
	transmission_weights(2,1) = 0;    // to stop
	transmission_weights(2,2) = w[0]; // to negative
	transmission_weights(2,3) = w[1]; // to positive
	// From positive
	transmission_weights(3,1) = 0;    // to stop
	transmission_weights(3,2) = w[3]; // to positive
	transmission_weights(3,3) = w[2]; // to negative
}

void CTwoStateModel::weights_to_vector(
    SGVector<float64_t>& psi, SGMatrix<float64_t> transmission_weights,
    SGVector<float64_t> emission_weights, index_t num_feats,
    index_t num_obs) const
{
	// Legend for state indices:
	// 0 -> start state
	// 1 -> stop state
	// 2 -> negative state
	// 3 -> positive state
	psi[0] = transmission_weights(2,2);
	psi[1] = transmission_weights(2,3);
	psi[2] = transmission_weights(3,3);
	psi[3] = transmission_weights(3,2);

	// start and stop states have no emission scores
	index_t obs_idx, psi_idx = m_num_transmission_params;
	for (index_t s = 2; s < m_num_states; ++s)
	{
		for (index_t f = 0; f < num_feats; ++f)
		{
			for (index_t o = 0; o < num_obs; ++o)
			{
				obs_idx = s*num_feats*num_obs + f*num_obs + o;
				psi[psi_idx++] = emission_weights[obs_idx];
			}
		}
	}

}

SGVector<float64_t> CTwoStateModel::weights_to_vector(
    SGMatrix<float64_t> transmission_weights,
    SGVector<float64_t> emission_weights, index_t num_feats,
    index_t num_obs) const
{
	index_t num_free_states = 2;
	SGVector< float64_t > vec(num_free_states*(num_free_states + num_feats*num_obs));
	vec.zero();
	weights_to_vector(vec, transmission_weights, emission_weights, num_feats, num_obs);
	return vec;
}

SGVector<index_t> CTwoStateModel::get_monotonicity(
    index_t num_free_states, index_t num_feats) const
{
	REQUIRE(num_free_states == 2, "Using the TwoStateModel only two states are free\n")

	SGVector<index_t> monotonicity(num_feats * num_free_states);

	for (index_t i = 0; i < num_feats; ++i)
		monotonicity[i] = -1;
	for (index_t i = num_feats; i < 2 * num_feats; ++i)
		monotonicity[i] = +1;

	return monotonicity;
}

CHMSVMModel* CTwoStateModel::simulate_data(
    index_t num_exm, index_t exm_len, index_t num_features,
    index_t num_noise_features)
{
	// Number of different states
	index_t num_states = 2;
	// Min and max length of positive block
	index_t block_len[] = {10, 100};
	// Min and max number of positive blocks per example
	index_t num_blocks[] = {0, 3};

	// Proportion of wrong labels
	float64_t prop_distort = 0.2;
	// Standard deviation of Gaussian noise
	float64_t noise_std = 4;

	// Generate label sequence randomly containing from num_blocks[0] to
	// num_blocks[1] blocks of positive labels each of length between
	// block_len[0] and block_len[1]

	CSequenceLabels* labels = new CSequenceLabels(num_exm, num_states);
	SGVector<index_t> ll(num_exm * exm_len);
	ll.zero();
	index_t rnb, rl, rp;

	for (index_t i = 0; i < num_exm; ++i)
	{
		SGVector<index_t> lab(exm_len);
		lab.zero();
		rnb = num_blocks[0] + CMath::ceil((num_blocks[1]-num_blocks[0])*
			CMath::random(0.0, 1.0)) - 1;

		for (index_t j = 0; j < rnb; ++j)
		{
			rl = block_len[0] + CMath::ceil((block_len[1]-block_len[0])*
				CMath::random(0.0, 1.0)) - 1;
			rp = CMath::ceil((exm_len-rl)*CMath::random(0.0, 1.0));

			for (index_t idx = rp - 1; idx < rp + rl; ++idx)
			{
				lab[idx] = 1;
				ll[i*exm_len + idx] = 1;
			}
		}

		labels->add_vector_label(lab);
	}

	// Generate features by
	// i) introducing label noise, i.e. flipping a propotion prop_distort
	// of labels and
	// ii) adding Gaussian noise to the (distorted) label sequence

	SGVector<index_t> distort(num_exm * exm_len);
	SGVector<index_t> d1(CMath::round(distort.vlen * prop_distort));
	SGVector<index_t> d2(d1.vlen);
	SGVector<index_t> lf;
	SGMatrix< float64_t > signal(num_features, distort.vlen);

	distort.range_fill();
	for (index_t i = 0; i < num_features; ++i)
	{
		lf = ll;
		CMath::permute(distort);

		for (index_t j = 0; j < d1.vlen; ++j)
			d1[j] = distort[j];

		for (index_t j = 0; j < d2.vlen; ++j)
			d2[j] = distort[ distort.vlen-d2.vlen+j ];

		for (index_t j = 0; j < d1.vlen; ++j)
			lf[ d1[j] ] = lf[ d2[j] ];

		index_t idx = i * signal.num_cols;
		for (index_t j = 0; j < signal.num_cols; ++j)
			signal[idx++] = lf[j] + noise_std*CMath::normal_random((float64_t)0.0, 1.0);
	}

	// Substitute some features by pure noise
	for (index_t i = 0; i < num_noise_features; ++i)
	{
		index_t idx = i * signal.num_cols;
		for (index_t j = 0; j < signal.num_cols; ++j)
			signal[idx++] = noise_std*CMath::normal_random((float64_t)0.0, 1.0);
	}

	CMatrixFeatures< float64_t >* features =
		new CMatrixFeatures< float64_t >(signal, exm_len, num_exm);

	index_t num_obs = 0; // continuous observations, dummy value
	bool use_plifs = true;
	return new CHMSVMModel(features, labels, SMT_TWO_STATE, num_obs, use_plifs);
}
