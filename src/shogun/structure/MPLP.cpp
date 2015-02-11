/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Jiaolong Xu
 * Copyright (C) 2014 Jiaolong Xu
 */

#include <shogun/structure/MPLP.h>
#include <shogun/io/SGIO.h>
#include <algorithm>

using namespace shogun;
using namespace std;

CEdgeFactor::CEdgeFactor(CFactor* factor, int32_t ind_factor)
{
	m_ind_factor = ind_factor;
	m_var_index = factor->get_variables();
	SGVector<int32_t> cards = factor->get_cardinalities();

	// Find the indices of each varibale within the edge.
	// Also intialize the message into that varSet.
	for (int32_t vi = 0; vi < m_var_index.size(); vi++)
	{
		m_inds_on_edge.push_back(vector<int32_t> (1, vi));

		// This will initialize the message and  set it to zero
		SGVector<int32_t> vec_cards(1);
		vec_cards[0] = cards[vi];
		SGNDArray<float64_t> curr_msg(vec_cards);
		curr_msg.set_const(0);
		m_msgs_from_edge.push_back(curr_msg);
	}
}

void CEdgeFactor::update_messages(vector<SGNDArray<float64_t> > &msgs_all_factors)
{
	// First do the expansion:
	// 1. Take out the message from the variable
	// 2. Expand it to the size of the edge engergy table
	// 3. Add the expanded message to the edge energy table
	// Set this to be the edge's variable set value
	SGNDArray<float64_t> orig_energy = msgs_all_factors[m_ind_factor].clone();

	for (int32_t vi = 0; vi < m_var_index.size(); vi++)
	{
		// Take out previous message
		SGVector<int32_t> inds_on_edge(1);
		inds_on_edge[0] = vi;
		SGNDArray<float64_t> arr_expand(orig_energy.get_dimensions());
		m_msgs_from_edge[vi].expand(arr_expand, inds_on_edge);
		orig_energy += arr_expand;
	}

	// Will store the total messages going into the intersection, but not from the edge
	vector<SGNDArray<float64_t> > lam_minus_edge;
	for (int32_t vi = 0; vi < m_var_index.size(); vi++)
	{
		int32_t curr_var = m_var_index[vi];
		SGNDArray<float64_t> curr_array = msgs_all_factors[curr_var].clone();
		lam_minus_edge.push_back(curr_array);
		lam_minus_edge.back() -= m_msgs_from_edge[vi];

		// For EMPLP: \lambda_j^{-i}(x_j) + \theta_{i,j}(x_i, x_j)
		// For GEMPLP: \sum_{\bar{s} \in S(c) \setminus s}\lambda_{\bar{s}}^{-c} + \theta_c(x_c)
		SGVector<int32_t> inds_on_edge(1);
		inds_on_edge[0] = vi;
		SGNDArray<float64_t> arr_expand(msgs_all_factors[m_ind_factor].get_dimensions());
		msgs_all_factors[curr_var].expand(arr_expand, inds_on_edge);
		msgs_all_factors[m_ind_factor] += arr_expand;
	}

	// Update messages
	max_into_multiple_subsets(msgs_all_factors[m_ind_factor], m_inds_on_edge, m_msgs_from_edge);
	int32_t sC = m_var_index.size();

	for (int32_t vi = 0; vi < m_var_index.size(); vi++)
	{
		// Take out previous message
		int32_t curr_var = m_var_index[vi];
		// Update message
		m_msgs_from_edge[vi] *= 1.0 / sC;
		// Put in current message
		msgs_all_factors[curr_var] = m_msgs_from_edge[vi].clone();
		// Finish updating message
		// msg_new = new - old + msg_old
		m_msgs_from_edge[vi] -= lam_minus_edge[vi];
		// Update original edge messages
		SGVector<int32_t> inds_on_edge(1);
		inds_on_edge[0] = vi;
		SGNDArray<float64_t> arr_expand(orig_energy.get_dimensions());
		m_msgs_from_edge[vi].expand(arr_expand, inds_on_edge);
		orig_energy -= arr_expand;

	}
	
	// update edge messages
	msgs_all_factors[m_ind_factor] = orig_energy.clone();

	return;
}

void CEdgeFactor::max_into_multiple_subsets(SGNDArray<float64_t> big_arr, vector<vector<int32_t> > &all_subset_inds, vector<SGNDArray<float64_t> > &all_max_res) const
{
	uint32_t nSubsets = all_subset_inds.size();
	bool* b_need_max = new bool[nSubsets];

	for (uint32_t si = 0; si < nSubsets; si++)
	{
		// If the subset equals the big array then maximizing would
		// give us the subset (assuming there is no reordering)
		if (all_subset_inds[si].size() == (uint32_t) big_arr.num_dims)
		{
			all_max_res[si] = big_arr.clone();
			b_need_max[si] = 0;
		}
		else
		{
			all_max_res[si].set_const(-1e9);
			b_need_max[si] = 1;
		}
	}

	// Go over all values of the big_array. For each check if its
	// value on the subset is larger than the current max
	SGVector<int32_t> inds_for_big(big_arr.num_dims);
	inds_for_big.zero();

	for (int32_t vi = 0; vi < big_arr.len_array; vi++)
	{
		for (uint32_t si = 0; si < nSubsets; si++)
		{
			if (!b_need_max[si])
				continue;
			
			int32_t y = 0;
			vector<int32_t> dim_in_big = all_subset_inds[si];
			
			if (dim_in_big.size() == 1)
			{
				y = inds_for_big[dim_in_big[0]];
			}
			else if (dim_in_big.size() == 2)
			{
				int32_t ind1 = dim_in_big[0];
				int32_t ind2 = dim_in_big[1];
				y = inds_for_big[ind1] * all_max_res[si].dims[1] + inds_for_big[ind2];
			}
			all_max_res[si][y] = max(all_max_res[si][y], big_arr.array[vi]);
		}
		big_arr.next_index(inds_for_big);
	}

	delete [] b_need_max;
}

CMPLP::CMPLP()
	: CMAPInferImpl()
{
	SG_UNSTABLE("CMPLP::CMPLP()", "\n");

	init();
}

CMPLP::CMPLP(CFactorGraph* fg, Parameter param)
	: CMAPInferImpl(fg),
	  m_param(param),
	  m_best_val(-CMath::INFTY),
	  m_last_obj(CMath::INFTY),
	  m_obj_del(CMath::INFTY)

{
	ASSERT(m_fg != NULL);

	init();
}

CMPLP::~CMPLP()
{

}

void CMPLP::init()
{
	// Set the variable sets to be all single nodes and also all edges
	// Initialize messages of all factors
	// First, add all individual nodes as their own variable set
	SGVector<int32_t> fg_var_sizes = m_fg->get_cardinalities();

	for (int32_t vi = 0; vi < fg_var_sizes.size(); vi++)
	{
		m_all_varSets.push_back(vector<int32_t>(1, vi));
		// Initialize messages (energies) of unary factors
		SGVector<int32_t> subset_size(1);
		subset_size[0] = fg_var_sizes[vi];
		SGNDArray<float64_t> curr_msg(subset_size);
		curr_msg.set_const(0);
		m_msgs_all_factors.push_back(curr_msg);
	}

	// Next initialize all edges. If not a unary factor, give them their own variable set
	CDynamicObjectArray* fg_factors = m_fg->get_factors();

	for (int32_t fi = 0; fi < fg_factors->get_num_elements(); fi++)
	{
		CFactor* factor = dynamic_cast<CFactor*>(fg_factors->get_element(fi));
		SGVector<int32_t> edge_vars = factor->get_variables();

		SGVector<float64_t> edge_energies = factor->get_energies();
		vector<float64_t> edge_messages(edge_energies.size());
		SGVector<int32_t> edge_cards = factor->get_cardinalities();

		// the defualt energy table index is y-x, we need to change it into x-y
		if (edge_cards.size() == 1)
		{
			for (uint32_t mi = 0; mi < edge_messages.size(); mi++)
				edge_messages[mi] = -edge_energies[mi];
		}
		else if (edge_cards.size() == 2)
		{
			for (int32_t x = 0; x < edge_cards[0]; x++)
			{
				for (int32_t y = 0; y < edge_cards[1]; y++)
				{
					float64_t energy = edge_energies[x * edge_cards[1] + y];
					// MPLP maximizes energy function
					edge_messages[y * edge_cards[0] + x] = - energy;
				}
			}
		}
		else if (edge_cards.size() > 2)
			SG_ERROR("Index issue has not been solved for higher order factors.");

		if (edge_vars.size() == 0)
			SG_DEBUG("Length zero edge!\n");

		if (edge_vars.size() == 1 && edge_messages.size() != 0)
		{
			SGNDArray<float64_t> curr_msg(edge_cards);

			for (int32_t i = 0; i < curr_msg.len_array; i++)
				curr_msg[i] = edge_messages[i];

			// Insert the unary potential into msgs_all_factors
			m_msgs_all_factors[edge_vars[0]] += curr_msg;
		}
		else // pairsie or higher order factors
		{
			vector<int32_t> curr_var_set(edge_vars.vector, edge_vars.vector + edge_vars.size());
			m_all_varSets.push_back(curr_var_set);

			int32_t curr_var_set_loc = m_all_varSets.size() - 1;

			if (edge_messages.size() != 0)
			{
				SGNDArray<float64_t> curr_msg(edge_cards);
				
				// Assume all_energies is given as a "flat" vector and put it into curr_lambda
				for (int32_t i = 0; i < curr_msg.len_array; i++)
					curr_msg[i] = edge_messages[i];
				
				m_msgs_all_factors.push_back(curr_msg);
				CEdgeFactor curr_edge(factor, curr_var_set_loc);
				m_all_edges.push_back(curr_edge);
			}
			else // Empty constructor
			{
				CEdgeFactor curr_edge(factor, curr_var_set_loc);
				m_all_edges.push_back(curr_edge);
			}
		}
		SG_UNREF(factor);
	}// iteration of factors
	SG_UNREF(fg_factors);

	// Initialize output vector
	m_assignment = SGVector<int32_t>(fg_var_sizes.size());
	m_best_assignment = SGVector<int32_t>(fg_var_sizes.size());
	m_assignment.zero();
	m_best_assignment.zero();
}

float64_t CMPLP::inference(SGVector<int32_t> assignment)
{
	REQUIRE(assignment.size() == m_fg->get_cardinalities().size(),
	        "%s::inference(): the output assignment should be prepared as"
	        "the same size as variables!\n", get_name());

	int32_t niter = m_param.m_max_iter;
	float64_t obj_del_thr = m_param.m_obj_del_thr;
	float64_t int_gap_thr = m_param.m_int_gap_thr;

	SG_SDEBUG("Initially running MPLP for %d iterations\n", niter);
	// run MPLP
	run_mplp(niter, obj_del_thr, int_gap_thr);

	for (int32_t vi = 0; vi < assignment.size(); vi++)
		assignment[vi] = m_best_assignment[vi];
	
	float64_t energy = m_fg->evaluate_energy(assignment);
	SG_DEBUG("fg.evaluate_energy(assignment) = %f\n", energy);

	return energy;
}

void CMPLP::run_mplp(int32_t niter, float64_t obj_del_thr, float64_t int_gap_thr)
{
	// block coordinate desent, outer loop
	for (int32_t it = 0; it < niter; ++it)
	{
		// inner loop, iterate over all edges
		for (uint32_t ei = 0; ei < m_all_edges.size(); ei++)
		{
			// fix others, optimize edge ei
			m_all_edges[ei].update_messages(m_msgs_all_factors);
		}

		float64_t obj;
		float64_t int_gap;

		// local decode, get the current assignment
		// and update the best assignment
		// compute the objective value
		obj = get_assignment();
		m_obj_del = m_last_obj - obj;
		m_last_obj = obj;

		int_gap = obj - m_best_val;

		SG_SDEBUG("Iter= %d Objective=%f ObjBest=%f ObjDel=%f Gap=%f \n", (it + 1), obj, m_best_val, m_obj_del, int_gap);

		if (m_obj_del < obj_del_thr && it > m_param.m_max_iter_inner)
			break;
		
		if (int_gap < int_gap_thr)
			break;
	}
	
	return;
}

float64_t CMPLP::get_assignment()
{
	float64_t obj = 0;
	int32_t max_at;

	for (uint32_t fi = 0; fi < m_msgs_all_factors.size(); fi++)
	{
		obj += m_msgs_all_factors[fi].max_element(max_at);

		if (m_all_varSets[fi].size() == 1)
			m_assignment[m_all_varSets[fi][0]] = max_at;
	}

	// Update best assignment
	float64_t curr_val = evaluate_assignment(m_assignment);
	if (curr_val > m_best_val)
	{
		SG_SDEBUG("Update best assignment, current value: %f\n", curr_val);
		m_best_assignment = m_assignment.clone();
		m_best_val = curr_val;
	}

	return obj;
}

float64_t CMPLP::evaluate_assignment(SGVector<int32_t> &assignment) const
{
	float64_t int_val = 0;
	SGVector<int32_t> fg_var_sizes = m_fg->get_cardinalities();
	uint32_t num_var = fg_var_sizes.size();
	
	// Iterates over edge factors
	for (uint32_t ri = 0; ri < m_all_edges.size(); ri++)
	{
		if (m_msgs_all_factors[ri+num_var].num_dims > 0)
		{
			// the index of the corresponding assignment in the message table
			SGVector<int32_t> var_index = m_all_edges[ri].get_variables();	
			SGVector<int32_t> tmpvec(var_index.size());
			
			for (int32_t vi = 0; vi < var_index.size(); vi++)
				tmpvec[vi] = assignment[var_index[vi]];
			
			int_val += m_msgs_all_factors[ri+num_var].get_value(tmpvec);
		}
	}

	//This iterates over all unary factors
	for (int32_t ni = 0; ni < assignment.size(); ni++)
		int_val += m_msgs_all_factors[ni][assignment[ni]];
	
	return int_val;
}
