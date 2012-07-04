/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/structure/HMSVMModel.h>
#include <shogun/features/MatrixFeatures.h>

using namespace shogun;

CHMSVMModel::CHMSVMModel()
: CStructuredModel()
{
	init();
}

CHMSVMModel::CHMSVMModel(CFeatures* features, CStructuredLabels* labels)
: CStructuredModel(features, labels)
{
	init();
}

CHMSVMModel::~CHMSVMModel()
{
}

int32_t CHMSVMModel::get_dim() const
{
	// Shorthand for the number of states
	int32_t S = ((CHMSVMLabels*) m_labels)->get_num_states();
	// Shorthand for the number of features of the feature vector
	int32_t D = ((CMatrixFeatures< float64_t >*) m_features)->get_num_features();

	return S*(S+D);
}

SGVector< float64_t > CHMSVMModel::get_joint_feature_vector(
		int32_t feat_idx,
		CStructuredData* y)
{
	// Shorthand for the number of states
	int32_t S = ((CHMSVMLabels*) m_labels)->get_num_states();
	// Shorthand for the number of features of the feature vector
	int32_t D = ((CMatrixFeatures< float64_t >*) m_features)->get_num_features();

	// Get the sequence of states sequence
	CSequence* yseq = CSequence::obtain_from_generic(y);
	// Ensure that length of the sequence of states and the one of the features
	// of x are the same
	ASSERT( ((CMatrixFeatures< float64_t >*) m_features)->
			get_feature_vector(feat_idx).num_cols == yseq->data.vlen);

	// Initialize psi
	SGVector< float64_t > psi( S*(S+D) );
	psi.zero();

	// TODO(just to prove correctness - to be removed when checked)
	ASSERT( psi.vlen == get_dim() );

	SGVector< float64_t > x_i(D);
	for ( int32_t i = 0 ; i < yseq->data.vlen ; ++i )
	{
		((CMatrixFeatures< float64_t >*) m_features)->
			get_feature_vector_col(x_i, feat_idx, i);

		add_transmission(psi.vector, yseq, i, S);
		add_emission(psi.vector+S*S, x_i.vector, yseq, i, D);
	}

	return psi;
}

void CHMSVMModel::add_transmission(
		float64_t* psi_trans,
		CSequence* y,
		int32_t i,
		int32_t S) const
{
	int32_t cur_state = y->data[i];
	int32_t prev_state;
	if ( i == 0 )
		prev_state = m_p[0];
	else
		prev_state = y->data[i-1];

	psi_trans[prev_state*S + cur_state] += 1;
}

void CHMSVMModel::add_emission(
		float64_t* psi_em,
		float64_t* x_i,
		CSequence* y,
		int32_t i,
		int32_t D) const
{
	// Obtain start, the index of the first element in psi_em to update
	int32_t cur_state = y->data[i];
	int32_t start     = cur_state*D;

	for ( int32_t j = 0 ; j < D ; ++j )
		psi_em[start++] += x_i[j];
}

CResultSet* CHMSVMModel::argmax(SGVector< float64_t > w, int32_t feat_idx)
{
	// Shorthand for the number of states
	int32_t S = ((CHMSVMLabels*) m_labels)->get_num_states();
	// Shorthand for the number of features of the feature vector
	int32_t D = ((CMatrixFeatures< float64_t >*) m_features)->get_num_features();

	ASSERT( w.vlen == S*(S+D) );

	// Compute the loss-augmented emission matrix:
	// (E)s,i = Delta(y_i, s) + w^em_s*x_i

	CSequence* y = CSequence::obtain_from_generic( m_labels->get_label(feat_idx) );
	int32_t T = y->data.vlen;
	SGMatrix< float64_t > E(S, T);
	SGVector< float64_t > x_i(D);

	// One element sequences for the state and an element of y
	CSequence* seq1 = new CSequence( SGVector< int32_t >(1) );
	CSequence* seq2 = new CSequence( SGVector< int32_t >(1) );

	float64_t partial_score;
	float64_t* w_em = w.vector + S*S;

	for ( int32_t i = 0 ; i < T ; ++i )
	{
		((CMatrixFeatures< float64_t >*) m_features)->
			get_feature_vector_col(x_i, feat_idx, i);

		for ( int32_t s = 0 ; s < S ; ++s )
		{
			partial_score = SGVector< float64_t >::dot(
					w_em + s*D, x_i.vector, D);

			seq1->data[0] = s;
			seq2->data[0] = y->data[i];

			E[s*T + i] = partial_score + delta_loss(seq1, seq2);
		}
	}

	// Initialize the dynamic programming table and the traceback matrix
	SGMatrix< float64_t >  dp(T, S);
	SGMatrix< float64_t > trb(T, S);

	for ( int32_t s = 0 ; s < S ; ++s )
	{
		if ( m_p[s] > -CMath::INFTY )
			dp[s] = E[s];
		else
			dp[s] = -CMath::INFTY;
	}

	// Viterbi algorithm
	int32_t idx;
	float64_t aij, tmp_score;

	for ( int32_t i = 1 ; i < T ; ++i )
		for ( int32_t cur = 0 ; cur < S ; ++cur )
		{
			idx = i*S + cur;

			 dp[idx] = -CMath::INFTY;
			trb[idx] = -1;

			for ( int32_t prev = 0 ; prev < S ; ++prev )
			{
				aij = w[cur*S + prev];

				if ( aij > -CMath::INFTY )
				{
					tmp_score = E[idx] + aij + dp[(i-1)*S + prev];

					if ( tmp_score > dp[idx] )
					{
						 dp[idx] = tmp_score;
						trb[idx] = prev;
					}
				}
			}
		}

	// Trace back the most likely sequence of states
	SGVector< int32_t > opt_path(T);
	CResultSet* ret = new CResultSet();
	SG_REF(ret);
	ret->score = -CMath::INFTY;
	opt_path[T-1] = -1;

	for ( int32_t s = 0 ; s < S ; ++s )
	{
		idx = (T-1)*S + s;

		if ( m_q[s] > -CMath::INFTY && dp[idx] > ret->score )
		{
			ret->score = dp[idx];
			opt_path[T-1] = s;
		}
	}

	for ( int32_t i = T-1 ; i > 0 ; --i )
	{
		opt_path[i-1] = trb[i*S + opt_path[i]];
	}

	// Populate the CResultSet object to return
	CSequence* ypred = new CSequence(opt_path);
	SG_REF(y);

	ret->psi_truth = CStructuredModel::get_joint_feature_vector(
			feat_idx, feat_idx);
	ret->psi_pred  = get_joint_feature_vector(feat_idx, ypred);
	ret->delta     = CStructuredModel::delta_loss(feat_idx, ypred);
	ret->argmax    = ypred;

	return ret;
}

float64_t CHMSVMModel::delta_loss(CStructuredData* y1, CStructuredData* y2)
{
	CSequence* seq1 = CSequence::obtain_from_generic(y1);
	CSequence* seq2 = CSequence::obtain_from_generic(y2);

	// Compute the Hamming loss, number of distinct elements in the sequences
	ASSERT( seq1->data.vlen == seq2->data.vlen );

	float64_t out = 0.0;
	for ( int32_t i = 0 ; i < seq1->data.vlen ; ++i )
		out += seq1->data[i] != seq2->data[i];

	return out;
}

/* TODO */
void CHMSVMModel::init_opt(
		SGMatrix< float64_t > A,
		SGVector< float64_t > a,
		SGMatrix< float64_t > B,
		SGVector< float64_t > b,
		SGVector< float64_t > lb,
		SGVector< float64_t > ub,
		SGMatrix< float64_t > & C)
{
}

bool CHMSVMModel::check_training_setup() const
{
	// Shorthand for the labels in the correct type
	CHMSVMLabels* hmsvm_labels = (CHMSVMLabels*) m_labels;
	// Frequency of each state
	SGVector< int32_t > state_freq( hmsvm_labels->get_num_states() );
	state_freq.zero();

	CSequence* seq;
	int32_t state;
	for ( int32_t i = 0 ; i < hmsvm_labels->get_num_labels() ; ++i )
	{
		seq = CSequence::obtain_from_generic(hmsvm_labels->get_label(i));

		for ( int32_t j = 0 ; j < seq->data.vlen ; ++j )
		{
			state = seq->data[j];

			if ( state < 0 || state >= hmsvm_labels->get_num_states() )
			{
				SG_ERROR("Found state out of {0, 1, ..., "
					 "num_states-1}\n");
				return false;
			}
			else
			{
				++state_freq[state];
			}
		}
	}

	for ( int32_t i = 0 ; i < hmsvm_labels->get_num_states() ; ++i )
	{
		if ( state_freq[i] <= 0 )
		{
			SG_ERROR("What? State %d has never appeared\n", i);
			return false;
		}
	}

	return true;
}

void CHMSVMModel::init()
{
	SG_ADD(&m_p, "m_p", "Distribution of start states", MS_NOT_AVAILABLE);
	SG_ADD(&m_q, "m_q", "Distribution of end states", MS_NOT_AVAILABLE);
}

CHMSVMModel* CHMSVMModel::simulate_two_state_model()
{
	// Number of examples
	int32_t num_exm = 1000;
	// Length of each example sequence
	int32_t exm_len = 250;
	// Number of different states
	int32_t num_states = 2;
	// Total number of features
	int32_t num_features = 10;
	// Number of features to be pure noise
	int32_t num_noise_features = 3;
	// Min and max length of positive block
	int32_t block_len[] = {10, 100};
	// Min and max number of positive blocks per example
	int32_t num_blocks[] = {0, 6};
	// Number of subsets for cross-validation
	int32_t num_subsets = 5;

	// Proportion of wrong labels
	float64_t prop_distort = 0.2;
	// Standard deviation of Gaussian noise
	float64_t noise_std = 4;

	// Generate label sequence randomly containing from num_blocks[0] to
	// num_blocks[1] blocks of positive labels each of length between
	// block_len[0] and block_len[1]

	CHMSVMLabels* labels = new CHMSVMLabels(num_exm, num_states);
	int32_t rnb, rl, rp;

	for ( int32_t i = 0 ; i < num_exm ; ++i)
	{
		SGVector< int32_t > l(exm_len);
		l.set_const(0);
		rnb = num_blocks[0] + CMath::ceil((num_blocks[1]-num_blocks[0])*
			CMath::random(0.0, 1.0)) - 1;

		for ( int32_t j = 0 ; j < rnb ; ++j )
		{
			rl = block_len[0] + CMath::ceil((block_len[1]-block_len[0])*
				CMath::random(0.0, 1.0)) - 1;
			rp = CMath::ceil((exm_len-rl)*CMath::random(0.0, 1.0));

			for ( int32_t idx = rp ; idx < rp+rl ; ++idx )
				l[idx] = 1;
		}

		labels->add_label(l);
	}

	// Generate features by
	// i) introducing label noise, i.e. flipping a propotion prop_distort
	// of labels and
	// ii) adding Gaussian noise to the (distorted) label sequence

	SGVector< int32_t >   distort(num_exm*exm_len);
	SGVector< int32_t >   d1(CMath::round(distort.vlen*prop_distort));
	SGVector< int32_t >   d2(d1.vlen);
	SGVector< float64_t > l = labels->to_double_vector();
	SGVector< float64_t > random(distort.vlen);

	CMatrixFeatures< float64_t >* features =
		new CMatrixFeatures< float64_t >(num_exm, num_features);

	for ( int32_t i = 0 ; i < num_features ; ++i )
	{
		distort.randperm();

		for ( int j = 0 ; j < d1.vlen ; ++j )
			d1[j] = distort[j];

		for ( int j = 0 ; j < d2.vlen ; ++j )
			d2[j] = distort[ distort.vlen-d2.vlen+j ];

		for ( int j = 0 ; j < d1.vlen ; ++j )
			labels->set_element(d1[j], labels->get_element(d2[j]));

		random.normal_random();
		random.scale(noise_std);
		SGVector< float64_t > signal = l + random;
		// TODO store the feautres properly in the CMatrixFeatures
	}

	// Substitute some features by pure noise
	// TODO


	return NULL;
}
