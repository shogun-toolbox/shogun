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
#include <shogun/features/VLMatrixFeatures.h>

using namespace shogun;

CHMSVMModel::CHMSVMModel()
: CStructuredModel()
{
}

CHMSVMModel::CHMSVMModel(CFeatures* features, CStructuredLabels* labels)
: CStructuredModel(features, labels)
{
}

CHMSVMModel::~CHMSVMModel()
{
}

/* TODO */
int32_t CHMSVMModel::get_dim() const
{
	return 0;
}

SGVector< float64_t > CHMSVMModel::get_joint_feature_vector(
		int32_t feat_idx,
		CStructuredData* y)
{
	// Shorthand for the number of states
	int32_t S = ((CHMSVMLabels*) m_labels)->get_num_states();
	// Shorthand for the number of features of the feature vector x
	SGStringList< float64_t > x = ((CVLMatrixFeatures< float64_t >*)
			m_features)->get_feature_vector(feat_idx);
	int32_t D = x.num_strings;
	// Shorthand for the length of the states sequence y
	CSequence* yseq = CSequence::obtain_from_generic(y);
	int32_t T = yseq->data.vlen;

	// Initialize psi
	SGVector< float64_t > psi( S*(S+D) ); // substitute this for get_dim()?
	psi.zero();

	for ( int32_t i = 1 ; i < T ; ++i )
	{
		add_transmission(psi.vector, yseq, i, S);
		add_emission(psi.vector+S*S, x.strings[i], yseq, i, D);
	}

	return psi;
}

void CHMSVMModel::add_transmission(
		float64_t* psi_trans,
		CSequence* y,
		int32_t i,
		int32_t S) const
{
	int32_t cur_state  = y->data[i];
	int32_t prev_state = y->data[i-1];

	psi_trans[prev_state*S + cur_state] += 1;
}

void CHMSVMModel::add_emission(
		float64_t* psi_em,
		SGString< float64_t > x_i,
		CSequence* y,
		int32_t i,
		int32_t D) const
{
	// Obtain start, the index of the first element in psi_em to update
	int32_t cur_state = y->data[i];
	int32_t start     = cur_state*D;

	for ( int32_t j = 0 ; j < D ; ++j )
		psi_em[start++] += x_i.string[j];
}

/* TODO */
CResultSet* CHMSVMModel::argmax(SGVector< float64_t > w, int32_t feat_idx)
{
	return NULL;
}

float64_t CHMSVMModel::delta_loss(int32_t ytrue_idx, CStructuredData* ypred)
{
	if ( ytrue_idx < 0 || ytrue_idx >= m_labels->get_num_labels() )
		SG_ERROR("The label index must be inside [0, num_labels-1]\n");

	if ( ypred->get_structured_data_type() != SDT_SEQUENCE )
		SG_ERROR("ypred must be a CSequence\n");

	// Shorthand for ypred with the correct structured data type
	CSequence* yhat  = CSequence::obtain_from_generic(ypred);
	// The same for ytrue
	CSequence* ytrue = CSequence::obtain_from_generic(
			m_labels->get_label(ytrue_idx));

	// Compute the Hamming loss, number of distinct elements in the sequences
	ASSERT( yhat->data.vlen == ytrue->data.vlen );

	float64_t out = 0.0;
	for ( int32_t i = 0 ; i < yhat->data.vlen ; ++i )
		out += yhat->data[i] != ytrue->data[i];

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
