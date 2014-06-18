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
#include <shogun/structure/TwoStateModel.h>
#include <shogun/structure/Plif.h>

using namespace shogun;

CHMSVMModel::CHMSVMModel()
: CStructuredModel()
{
	init();
}

CHMSVMModel::CHMSVMModel(CFeatures* features, CStructuredLabels* labels, EStateModelType smt, int32_t num_obs, bool use_plifs)
: CStructuredModel(features, labels)
{
	init();
	m_num_obs = num_obs;
	m_num_plif_nodes = 20;
	m_use_plifs = use_plifs;

	switch (smt)
	{
		case SMT_TWO_STATE:
			m_state_model = new CTwoStateModel();
			break;
		case SMT_UNKNOWN:
		default:
			SG_ERROR("The EStateModelType given is not valid\n")
	}
}

CHMSVMModel::~CHMSVMModel()
{
	SG_UNREF(m_state_model);
	SG_UNREF(m_plif_matrix);
}

int32_t CHMSVMModel::get_dim() const
{
	// Shorthand for the number of free states
	int32_t free_states = ((CSequenceLabels*) m_labels)->get_num_states();
	CMatrixFeatures< float64_t >* mf = (CMatrixFeatures< float64_t >*) m_features;
	int32_t D = mf->get_num_features();

	if ( m_use_plifs )
		return free_states*(free_states + D*m_num_plif_nodes);
	else
		return free_states*(free_states + D*m_num_obs);
}

SGVector< float64_t > CHMSVMModel::get_joint_feature_vector(
		int32_t feat_idx,
		CStructuredData* y)
{
	// Shorthand for the number of features of the feature vector
	CMatrixFeatures< float64_t >* mf = (CMatrixFeatures< float64_t >*) m_features;
	int32_t D = mf->get_num_features();

	// Get the sequence of labels
	CSequence* label_seq = CSequence::obtain_from_generic(y);

	// Initialize psi
	SGVector< float64_t > psi(get_dim());
	psi.zero();

	// Translate from labels sequence to state sequence
	SGVector< int32_t > state_seq = m_state_model->labels_to_states(label_seq);
	m_transmission_weights.zero();

	for ( int32_t i = 0 ; i < state_seq.vlen-1 ; ++i )
		m_transmission_weights(state_seq[i],state_seq[i+1]) += 1;

	SGMatrix< float64_t > obs = mf->get_feature_vector(feat_idx);
	REQUIRE(obs.num_rows == D && obs.num_cols == state_seq.vlen,
		"obs.num_rows (%d) != D (%d) OR obs.num_cols (%d) != state_seq.vlen (%d)\n",
		obs.num_rows, D, obs.num_cols, state_seq.vlen)
	m_emission_weights.zero();
	index_t aux_idx, weight_idx;

	if ( !m_use_plifs )	// Do not use PLiFs
	{
		for ( int32_t f = 0 ; f < D ; ++f )
		{
			aux_idx = f*m_num_obs;

			for ( int32_t j = 0 ; j < state_seq.vlen ; ++j )
			{
				weight_idx = aux_idx + state_seq[j]*D*m_num_obs + obs(f,j);
				m_emission_weights[weight_idx] += 1;
			}
		}

		m_state_model->weights_to_vector(psi, m_transmission_weights, m_emission_weights,
				D, m_num_obs);
	}
	else	// Use PLiFs
	{
		int32_t S = m_state_model->get_num_states();

		for ( int32_t f = 0 ; f < D ; ++f )
		{
			aux_idx = f*m_num_plif_nodes;

			for ( int32_t j = 0 ; j < state_seq.vlen ; ++j )
			{
				CPlif* plif = (CPlif*) m_plif_matrix->get_element(S*f + state_seq[j]);
				SGVector<float64_t> limits = plif->get_plif_limits();
				// The number of supporting points smaller or equal than value
				int32_t count = 0;
				// The observation value
				float64_t value = obs(f,j);

				for ( int32_t i = 0 ; i < m_num_plif_nodes ; ++i )
				{
					if ( limits[i] <= value )
						++count;
					else
						break;
				}

				weight_idx = aux_idx + state_seq[j]*D*m_num_plif_nodes;

				if ( count == 0 )
					m_emission_weights[weight_idx] += 1;
				else if ( count == m_num_plif_nodes )
					m_emission_weights[weight_idx + m_num_plif_nodes-1] += 1;
				else
				{
					m_emission_weights[weight_idx + count] +=
						(value-limits[count-1]) / (limits[count]-limits[count-1]);

					m_emission_weights[weight_idx + count-1] +=
						(limits[count]-value) / (limits[count]-limits[count-1]);
				}

				SG_UNREF(plif);
			}
		}

		m_state_model->weights_to_vector(psi, m_transmission_weights, m_emission_weights,
				D, m_num_plif_nodes);
	}

	return psi;
}

CResultSet* CHMSVMModel::argmax(
		SGVector< float64_t > w,
		int32_t feat_idx,
		bool const training)
{
	int32_t dim = get_dim();
	ASSERT( w.vlen == get_dim() )

	// Shorthand for the number of features of the feature vector
	CMatrixFeatures< float64_t >* mf = (CMatrixFeatures< float64_t >*) m_features;
	int32_t D = mf->get_num_features();
	// Shorthand for the number of states
	int32_t S = m_state_model->get_num_states();

	if ( m_use_plifs )
	{
		REQUIRE(m_plif_matrix, "PLiF matrix not allocated, has the SO machine been trained with "
				"the use_plifs option?\n");
		REQUIRE(m_plif_matrix->get_num_elements() == S*D, "Dimension mismatch in PLiF matrix, have the "
				"feature dimension and/or number of states changed from training to prediction?\n");
	}

	// Distribution of start states
	SGVector< float64_t > p = m_state_model->get_start_states();
	// Distribution of stop states
	SGVector< float64_t > q = m_state_model->get_stop_states();

	// Compute the loss-augmented emission matrix:
	// E_{s,i} = w^{em}_s \cdot x_i + Delta(y_i, s)

	SGMatrix< float64_t > x = mf->get_feature_vector(feat_idx);

	int32_t T = x.num_cols;
	SGMatrix< float64_t > E(S, T);
	E.zero();

	if ( !m_use_plifs )	// Do not use PLiFs
	{
		index_t em_idx;
		m_state_model->reshape_emission_params(m_emission_weights, w, D, m_num_obs);

		for ( int32_t i = 0 ; i < T ; ++i )
		{
			for ( int32_t j = 0 ; j < D ; ++j )
			{
				//FIXME make independent of observation values
				em_idx = j*m_num_obs + (index_t)CMath::round(x(j,i));

				for ( int32_t s = 0 ; s < S ; ++s )
					E(s,i) += m_emission_weights[s*D*m_num_obs + em_idx];
			}
		}
	}
	else	// Use PLiFs
	{
		m_state_model->reshape_emission_params(m_plif_matrix, w, D, m_num_plif_nodes);

		for ( int32_t i = 0 ; i < T ; ++i )
		{
			for ( int32_t f = 0 ; f < D ; ++f )
			{
				for ( int32_t s = 0 ; s < S ; ++s )
				{
					CPlif* plif = (CPlif*) m_plif_matrix->get_element(S*f + s);
					E(s,i) += plif->lookup( x(f,i) );
					SG_UNREF(plif);
				}
			}
		}
	}

	// If argmax used while training, add to E the loss matrix (loss-augmented inference)
	if ( training )
	{
		CSequence* ytrue =
			CSequence::obtain_from_generic(m_labels->get_label(feat_idx));

		REQUIRE(ytrue->get_data().size() == T, "T, the length of the feature "
			"x^i (%d) and the length of its corresponding label y^i "
			"(%d) must be the same.\n", T, ytrue->get_data().size());

		SGMatrix< float64_t > loss_matrix = m_state_model->loss_matrix(ytrue);

		ASSERT(loss_matrix.num_rows == E.num_rows &&
		       loss_matrix.num_cols == E.num_cols);

		SGVector< float64_t >::add(E.matrix, 1.0, E.matrix,
				1.0, loss_matrix.matrix, E.num_rows*E.num_cols);

		// Decrement the reference count corresponding to get_label above
		SG_UNREF(ytrue);
	}

	// Initialize the dynamic programming table and the traceback matrix
	SGMatrix< float64_t >  dp(T, S);
	SGMatrix< float64_t > trb(T, S);
	m_state_model->reshape_transmission_params(m_transmission_weights, w);

	for ( int32_t s = 0 ; s < S ; ++s )
	{
		if ( p[s] > -CMath::INFTY )
		{
			// dp(0,s) = E(s,0)
			dp(0,s) = E[s];
		}
		else
		{
			dp(0,s) = -CMath::INFTY;
		}
	}

	// Viterbi algorithm
	int32_t idx;
	float64_t tmp_score, e, a;

	for ( int32_t i = 1 ; i < T ; ++i )
	{
		for ( int32_t cur = 0 ; cur < S ; ++cur )
		{
			idx = cur*T + i;

			 dp[idx] = -CMath::INFTY;
			trb[idx] = -1;

			// e = E(cur,i)
			e = E[i*S + cur];

			for ( int32_t prev = 0 ; prev < S ; ++prev )
			{
				// aij = m_transmission_weights(prev, cur)
				a = m_transmission_weights[cur*S + prev];

				if ( a > -CMath::INFTY )
				{
					// tmp_score = e + a + dp(i-1, prev)
					tmp_score = e + a + dp[prev*T + i-1];

					if ( tmp_score > dp[idx] )
					{
						 dp[idx] = tmp_score;
						trb[idx] = prev;
					}
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
		idx = s*T + T-1;

		if ( q[s] > -CMath::INFTY && dp[idx] > ret->score )
		{
			ret->score = dp[idx];
			opt_path[T-1] = s;
		}
	}

	REQUIRE(opt_path[T-1]!=-1, "Viterbi decoding found no possible sequence states.\n"
			"Maybe the state model used cannot produce such sequence.\n"
			"If using the TwoStateModel, please use sequences of length greater than two.\n");

	for ( int32_t i = T-1 ; i > 0 ; --i )
		opt_path[i-1] = trb[opt_path[i]*T + i];

	// Populate the CResultSet object to return
	CSequence* ypred = m_state_model->states_to_labels(opt_path);

	ret->psi_pred = get_joint_feature_vector(feat_idx, ypred);
	ret->argmax   = ypred;
	if ( training )
	{
		ret->delta     = CStructuredModel::delta_loss(feat_idx, ypred);
		ret->psi_truth = CStructuredModel::get_joint_feature_vector(feat_idx, feat_idx);
		ret->score    -= SGVector< float64_t >::dot(w.vector, ret->psi_truth.vector, dim);
	}

	return ret;
}

float64_t CHMSVMModel::delta_loss(CStructuredData* y1, CStructuredData* y2)
{
	CSequence* seq1 = CSequence::obtain_from_generic(y1);
	CSequence* seq2 = CSequence::obtain_from_generic(y2);

	// Compute the Hamming loss, number of distinct elements in the sequences
	return m_state_model->loss(seq1, seq2);
}

void CHMSVMModel::init_primal_opt(
		float64_t regularization,
		SGMatrix< float64_t > & A,
		SGVector< float64_t > a,
		SGMatrix< float64_t > B,
		SGVector< float64_t > & b,
		SGVector< float64_t > lb,
		SGVector< float64_t > ub,
		SGMatrix< float64_t > & C)
{
	// Shorthand for the number of free states (i.e. states for which parameters are learnt)
	int32_t S = ((CSequenceLabels*) m_labels)->get_num_states();
	// Shorthand for the number of features of the feature vector
	int32_t D = ((CMatrixFeatures< float64_t >*) m_features)->get_num_features();

	// Monotonicity constraints for feature scoring functions
	SGVector< int32_t > monotonicity = m_state_model->get_monotonicity(S,D);

	// Quadratic regularization

	float64_t C_small = regularization;
	float64_t C_smooth = 0.02*regularization;
	// TODO change the representation of C to sparse matrix
	C = SGMatrix< float64_t >(get_dim()+m_num_aux, get_dim()+m_num_aux);
	C.zero();
	for ( int32_t i = 0 ; i < get_dim() ; ++i )
		C(i,i) = C_small;
	for ( int32_t i = get_dim() ; i < get_dim()+m_num_aux ; ++i )
		C(i,i) = C_smooth;

	// Smoothness constraints

	// For each auxiliary variable, there are two different constraints
	// TODO change the representation of A to sparse matrix
	A = SGMatrix< float64_t >(2*m_num_aux, get_dim()+m_num_aux);
	A.zero();

	// Indices to the beginning of the blocks of scores. Each block is
	// formed by the scores of a pair (state, feature)
	SGVector< int32_t > score_starts(S*D);
	int32_t delta = m_use_plifs ? m_num_plif_nodes : m_num_obs;
	for ( int32_t idx = S*S, k = 0 ; k < S*D ; idx += delta, ++k )
		score_starts[k] = idx;

	// Indices to the beginning of the blocks of variables for smoothness
	SGVector< int32_t > aux_starts_smooth(S*D);
	for ( int32_t idx = get_dim(), k = 0 ; k < S*D ; idx += delta-1, ++k )
		aux_starts_smooth[k] = idx;

	// Bound the difference between adjacent score values from above and
	// below by an auxiliary variable (which then is regularized
	// quadratically)

	int32_t con_idx = 0, scr_idx, aux_idx;

	for ( int32_t i = 0 ; i < score_starts.vlen ; ++i )
	{
		scr_idx = score_starts[i];
		aux_idx = aux_starts_smooth[i];

		for ( int32_t j = 0 ; j < delta-1 ; ++j )
		{
			A(con_idx, scr_idx)   =  1;
			A(con_idx, scr_idx+1) = -1;

			if ( monotonicity[i] != 1 )
				A(con_idx, aux_idx) = -1;
			++con_idx;

			A(con_idx, scr_idx)   = -1;
			A(con_idx, scr_idx+1) =  1;

			if ( monotonicity[i] != -1 )
				A(con_idx, aux_idx) = -1;
			++con_idx;

			++scr_idx, ++aux_idx;
		}
	}

	// Bounds for the smoothness constraints
	b = SGVector< float64_t >(2*m_num_aux);
	b.zero();
}

bool CHMSVMModel::check_training_setup() const
{
	// Shorthand for the labels in the correct type
	CSequenceLabels* hmsvm_labels = (CSequenceLabels*) m_labels;
	// Frequency of each state
	SGVector< int32_t > state_freq( hmsvm_labels->get_num_states() );
	state_freq.zero();

	CSequence* seq;
	int32_t state;
	for ( int32_t i = 0 ; i < hmsvm_labels->get_num_labels() ; ++i )
	{
		seq = CSequence::obtain_from_generic(hmsvm_labels->get_label(i));

		SGVector<int32_t> seq_data = seq->get_data();
		for ( int32_t j = 0 ; j < seq_data.size() ; ++j )
		{
			state = seq_data[j];

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

		// Decrement the reference count increased by get_label
		SG_UNREF(seq);
	}

	for ( int32_t i = 0 ; i < hmsvm_labels->get_num_states() ; ++i )
	{
		if ( state_freq[i] <= 0 )
		{
			SG_ERROR("What? State %d has never appeared\n", i)
			return false;
		}
	}

	return true;
}

void CHMSVMModel::init()
{
	SG_ADD((CSGObject**) &m_state_model, "m_state_model", "The state model", MS_NOT_AVAILABLE);
	SG_ADD(&m_transmission_weights, "m_transmission_weights",
			"Transmission weights used in Viterbi", MS_NOT_AVAILABLE);
	SG_ADD(&m_emission_weights, "m_emission_weights",
			"Emission weights used in Viterbi", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_plif_nodes, "m_num_plif_nodes", "The number of points per PLiF",
			MS_NOT_AVAILABLE); // FIXME It would actually make sense to do MS for this parameter
	SG_ADD(&m_use_plifs, "m_use_plifs", "Whether to use plifs", MS_NOT_AVAILABLE);

	m_num_obs = 0;
	m_num_aux = 0;
	m_use_plifs = false;
	m_state_model = NULL;
	m_plif_matrix = NULL;
	m_num_plif_nodes = 0;
}

int32_t CHMSVMModel::get_num_aux() const
{
	return m_num_aux;
}

int32_t CHMSVMModel::get_num_aux_con() const
{
	return 2*m_num_aux;
}

void CHMSVMModel::set_use_plifs(bool use_plifs)
{
	m_use_plifs = use_plifs;
}

void CHMSVMModel::init_training()
{
	// Shorthands for the number of states, the matrix features and their dimension
	int32_t S = m_state_model->get_num_states();
	CMatrixFeatures< float64_t >* mf = (CMatrixFeatures< float64_t >*) m_features;
	int32_t D = mf->get_num_features();

	// Transmission and emission weights allocation
	m_transmission_weights = SGMatrix< float64_t >(S,S);
	if ( m_use_plifs )
		m_emission_weights = SGVector< float64_t >(S*D*m_num_plif_nodes);
	else
		m_emission_weights = SGVector< float64_t >(S*D*m_num_obs);

	// Auxiliary variables

	// Shorthand for the number of free states
	int32_t free_states = ((CSequenceLabels*) m_labels)->get_num_states();
	if ( m_use_plifs )
		m_num_aux = free_states*D*(m_num_plif_nodes-1);
	else
		m_num_aux = free_states*D*(m_num_obs-1);

	if ( m_use_plifs )
	{
		// Initialize PLiF matrix
		m_plif_matrix = new CDynamicObjectArray(S*D);
		SG_REF(m_plif_matrix);

		// Determine the x values for the supporting points of the PLiFs

		// Count the number of points per feature, using all the feature vectors
		int32_t N = 0;
		for ( int32_t i = 0 ; i < mf->get_num_vectors() ; ++i )
		{
			SGMatrix<float64_t> feat_vec = mf->get_feature_vector(i);
			N += feat_vec.num_cols;
		}

		// Choose the supporting points so that roughly the same number of points fall in each bin
		SGVector< float64_t > a = SGVector< float64_t >::linspace_vec(1, N, m_num_plif_nodes+1);
		SGVector< index_t > signal_idxs(m_num_plif_nodes);
		for ( int32_t i = 0 ; i < signal_idxs.vlen ; ++i )
			signal_idxs[i] = (index_t) CMath::round( (a[i] + a[i+1]) / 2 ) - 1;

		SGVector< float64_t > signal(N);
		index_t idx; // used to populate signal
		for ( int32_t f = 0 ; f < D ; ++f )
		{
			// Get the points of feature f of all the feature vectors
			idx = 0;
			for ( int32_t i = 0 ; i < mf->get_num_vectors() ; ++i )
			{
				SGMatrix<float64_t> feat_vec = mf->get_feature_vector(i);
				for ( int32_t j = 0 ; j < feat_vec.num_cols ; ++j )
					signal[idx++] = feat_vec(f,j);
			}

			signal.qsort();
			SGVector< float64_t > limits(m_num_plif_nodes);
			for ( int32_t i = 0 ; i < m_num_plif_nodes ; ++i )
				limits[i] = signal[ signal_idxs[i] ];

			// Set the PLiFs' supporting points
			for ( int32_t s = 0 ; s < S ; ++s )
			{
				CPlif* plif = new CPlif(m_num_plif_nodes);
				plif->set_plif_limits(limits);
				plif->set_min_value(-CMath::INFTY);
				plif->set_max_value(CMath::INFTY);
				m_plif_matrix->push_back(plif);
			}
		}
	}
}

SGMatrix< float64_t > CHMSVMModel::get_transmission_weights() const
{
	return m_transmission_weights;
}

SGVector< float64_t > CHMSVMModel::get_emission_weights() const
{
	return m_emission_weights;
}

CStateModel* CHMSVMModel::get_state_model() const
{
	SG_REF(m_state_model);
	return m_state_model;
}
