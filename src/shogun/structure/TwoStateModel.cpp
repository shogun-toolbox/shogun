/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/structure/TwoStateModel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/MatrixFeatures.h>
#include <shogun/structure/Plif.h>
#include <shogun/mathematics/RandomNamespace.h>
#include <shogun/mathematics/UniformRealDistribution.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <random>

using namespace shogun;

TwoStateModel::TwoStateModel() : StateModel()
{
	// The number of states in this state model is equal to four.
	// Although parameters are learnt only for two of them, other
	// two states (start and stop) are used
	m_num_states = 4;
	m_num_transmission_params = 4;

	m_state_loss_mat = SGMatrix< float64_t >(m_num_states, m_num_states);
	m_state_loss_mat.zero();
	for ( int32_t i = 0 ; i < m_num_states-1 ; ++i )
	{
		m_state_loss_mat(m_num_states-1, i) = 1;
		m_state_loss_mat(i, m_num_states-1) = 1;
	}

	// Initialize the start and stop states
	m_p = SGVector< float64_t >(m_num_states);
	m_q = SGVector< float64_t >(m_num_states);
	m_p.set_const(-Math::INFTY);
	m_q.set_const(-Math::INFTY);
	m_p[0] = 0; // start state
	m_q[1] = 0; // stop  state
}

TwoStateModel::~TwoStateModel()
{
}

SGMatrix< float64_t > TwoStateModel::loss_matrix(std::shared_ptr<Sequence> label_seq)
{
	SGVector< int32_t > state_seq = labels_to_states(label_seq);
	SGMatrix< float64_t > loss_mat(m_num_states, state_seq.vlen);

	for ( int32_t i = 0 ; i < loss_mat.num_cols ; ++i )
	{
		for ( int32_t s = 0 ; s < loss_mat.num_rows ; ++s )
			loss_mat(s,i) = m_state_loss_mat(s, state_seq[i]);
	}

	return loss_mat;
}

float64_t TwoStateModel::loss(std::shared_ptr<Sequence> label_seq_lhs, std::shared_ptr<Sequence> label_seq_rhs)
{
	SGVector< int32_t > state_seq_lhs = labels_to_states(label_seq_lhs);
	SGVector< int32_t > state_seq_rhs = labels_to_states(label_seq_rhs);

	ASSERT(state_seq_lhs.vlen == state_seq_rhs.vlen)

	float64_t ret = 0.0;
	for ( int32_t i = 0 ; i < state_seq_lhs.vlen ; ++i )
		ret += m_state_loss_mat(state_seq_lhs[i], state_seq_rhs[i]);

	return ret;
}

SGVector< int32_t > TwoStateModel::labels_to_states(std::shared_ptr<Sequence> label_seq) const
{
	// 0 -> start state
	// 1 -> stop state
	// 2 -> negative state (label == 0)
	// 3 -> positive state (label == 1)

	SGVector< int32_t > seq_data = label_seq->get_data();
	SGVector< int32_t > state_seq(seq_data.size());
	for ( int32_t i = 1 ; i < state_seq.vlen-1 ; ++i )
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

std::shared_ptr<Sequence> TwoStateModel::states_to_labels(SGVector< int32_t > state_seq) const
{
	SGVector< int32_t > label_seq(state_seq.vlen);

	//FIXME make independent of values 0-1 in labels
	// Legend for state indices:
	// 0 -> start state => label 0
	// 1 -> stop state => label 0
	// 2 -> negative state (label == 0) => label 0
	// 3 -> positive state (label == 1) => label 1
	label_seq.zero();
	for ( int32_t i = 0 ; i < state_seq.vlen ; ++i )
	{
		if ( state_seq[i] == 3 )
			label_seq[i] = 1;
	}

	return std::make_shared<Sequence>(label_seq);
}

void TwoStateModel::reshape_emission_params(SGVector< float64_t >& emission_weights,
		SGVector< float64_t > w, int32_t num_feats, int32_t num_obs)
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
	for ( int32_t s = 2 ; s < m_num_states ; ++s )
	{
		for ( int32_t f = 0 ; f < num_feats ; ++f )
		{
			for ( int32_t o = 0 ; o < num_obs ; ++o )
			{
				em_idx = s*num_feats*num_obs + f*num_obs + o;
				emission_weights[em_idx] = w[w_idx++];
			}
		}
	}
}

void TwoStateModel::reshape_emission_params(const std::vector<std::shared_ptr<Plif>>& plif_matrix,
		SGVector< float64_t > w, int32_t num_feats, int32_t num_plif_nodes)
{
	index_t p_idx, w_idx = m_num_transmission_params;
	for ( int32_t s = 2 ; s < m_num_states ; ++s )
	{
		for ( int32_t f = 0 ; f < num_feats ; ++f )
		{
			SGVector< float64_t > penalties(num_plif_nodes);
			p_idx = 0;

			for ( int32_t i = 0 ; i < num_plif_nodes ; ++i )
				penalties[p_idx++] = w[w_idx++];

			auto plif = plif_matrix[m_num_states*f + s];
			plif->set_plif_penalty(penalties);

		}
	}
}

void TwoStateModel::reshape_transmission_params(
		SGMatrix< float64_t >& transmission_weights, SGVector< float64_t > w)
{
	transmission_weights.set_const(-Math::INFTY);

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

void TwoStateModel::weights_to_vector(SGVector< float64_t >& psi,
		SGMatrix< float64_t > transmission_weights,
		SGVector< float64_t > emission_weights,
		int32_t num_feats, int32_t num_obs) const
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
	for ( int32_t s = 2 ; s < m_num_states ; ++s )
	{
		for ( int32_t f = 0 ; f < num_feats ; ++f )
		{
			for ( int32_t o = 0 ; o < num_obs ; ++o )
			{
				obs_idx = s*num_feats*num_obs + f*num_obs + o;
				psi[psi_idx++] = emission_weights[obs_idx];
			}
		}
	}

}

SGVector< float64_t > TwoStateModel::weights_to_vector(SGMatrix< float64_t > transmission_weights,
		SGVector< float64_t > emission_weights, int32_t num_feats, int32_t num_obs) const
{
	int32_t num_free_states = 2;
	SGVector< float64_t > vec(num_free_states*(num_free_states + num_feats*num_obs));
	vec.zero();
	weights_to_vector(vec, transmission_weights, emission_weights, num_feats, num_obs);
	return vec;
}

SGVector< int32_t > TwoStateModel::get_monotonicity(int32_t num_free_states,
		int32_t num_feats) const
{
	require(num_free_states == 2, "Using the TwoStateModel only two states are free");

	SGVector< int32_t > monotonicity(num_feats*num_free_states);

	for ( int32_t i = 0 ; i < num_feats ; ++i )
		monotonicity[i] = -1;
	for ( int32_t i = num_feats ; i < 2*num_feats ; ++i )
		monotonicity[i] = +1;

	return monotonicity;
}

template <typename PRNG>
std::shared_ptr<HMSVMModel> TwoStateModel::simulate_data(int32_t num_exm, int32_t exm_len,
	int32_t num_features, int32_t num_noise_features, PRNG& prng)
{
	// Number of different states
	int32_t num_states = 2;
	// Min and max length of positive block
	int32_t block_len[] = {10, 100};
	// Min and max number of positive blocks per example
	int32_t num_blocks[] = {0, 3};

	// Proportion of wrong labels
	float64_t prop_distort = 0.2;
	// Standard deviation of Gaussian noise
	float64_t noise_std = 4;

	// Generate label sequence randomly containing from num_blocks[0] to
	// num_blocks[1] blocks of positive labels each of length between
	// block_len[0] and block_len[1]

	auto labels = std::make_shared<SequenceLabels>(num_exm, num_states);
	SGVector< int32_t > ll(num_exm*exm_len);
	ll.zero();
	int32_t rnb, rl, rp;
	UniformRealDistribution<float64_t> uniform_real_dist(0.0, 1.0);
	for ( int32_t i = 0 ; i < num_exm ; ++i)
	{
		SGVector< int32_t > lab(exm_len);
		lab.zero();
		rnb = num_blocks[0] +
		      std::ceil(
		          (num_blocks[1] - num_blocks[0]) * uniform_real_dist(prng)) -
		      1;

		for ( int32_t j = 0 ; j < rnb ; ++j )
		{
			rl = block_len[0] +
			     std::ceil(
			         (block_len[1] - block_len[0]) * uniform_real_dist(prng)) -
			     1;
			rp = std::ceil((exm_len - rl) * uniform_real_dist(prng));

			for ( int32_t idx = rp-1 ; idx < rp+rl ; ++idx )
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

	SGVector< int32_t >   distort(num_exm*exm_len);
	SGVector< int32_t >   d1(Math::round(distort.vlen*prop_distort));
	SGVector< int32_t >   d2(d1.vlen);
	SGVector< int32_t >   lf;
	SGMatrix< float64_t > signal(num_features, distort.vlen);
	NormalDistribution<float64_t> normal_dist;

	distort.range_fill();
	for ( int32_t i = 0 ; i < num_features ; ++i )
	{
		lf = ll;
		random::shuffle(distort, prng);

		for ( int32_t j = 0 ; j < d1.vlen ; ++j )
			d1[j] = distort[j];

		for ( int32_t j = 0 ; j < d2.vlen ; ++j )
			d2[j] = distort[ distort.vlen-d2.vlen+j ];

		for ( int32_t j = 0 ; j < d1.vlen ; ++j )
			lf[ d1[j] ] = lf[ d2[j] ];

		int32_t idx = i*signal.num_cols;
		for ( int32_t j = 0 ; j < signal.num_cols ; ++j )
			signal[idx++] = lf[j] + noise_std*normal_dist(prng);
	}

	// Substitute some features by pure noise
	for ( int32_t i = 0 ; i < num_noise_features ; ++i )
	{
		int32_t idx = i*signal.num_cols;
		for ( int32_t j = 0 ; j < signal.num_cols ; ++j )
			signal[idx++] = noise_std*normal_dist(prng);
	}

	auto features =
		std::make_shared<MatrixFeatures< float64_t >>(signal, exm_len, num_exm);

	int32_t num_obs = 0; // continuous observations, dummy value
	bool use_plifs = true;
	return std::make_shared<HMSVMModel>(features, labels, SMT_TWO_STATE, num_obs, use_plifs);
}

template std::shared_ptr<HMSVMModel> TwoStateModel::simulate_data<std::mt19937_64>(int32_t num_exm, int32_t exm_len,
	int32_t num_features, int32_t num_noise_features, std::mt19937_64& prng);

std::shared_ptr<HMSVMModel> TwoStateModel::simulate_data(int32_t num_exm, int32_t exm_len,
	int32_t num_features, int32_t num_noise_features, int32_t seed)
{
	std::mt19937_64 prng(seed);
	return simulate_data(num_exm, exm_len, num_features, num_noise_features, prng);
}
