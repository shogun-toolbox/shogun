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

CRegion::CRegion(vector<int32_t> region_inds, vector<vector<int32_t> > all_intersects, vector<int32_t> intersect_inds, vector<int32_t> var_sizes, int32_t region_intersect)
	: m_region_inds(region_inds), m_intersect_inds(intersect_inds), m_region_intersect(region_intersect)
{
	// Find the indices of each intersection within the region. Also intialize the message into that intersection
	for (uint32_t si = 0; si < m_intersect_inds.size(); ++si)
	{
		vector<int32_t> tmp_inds_of_intersects;
		vector<int32_t> curr_intersect = all_intersects[m_intersect_inds[si]];
		vector<int32_t> intersect_var_sizes;

		// Go over all variables in the intersection set
		for (uint32_t i = 0; i < curr_intersect.size(); ++i)
		{
			int32_t var_in_intersect = curr_intersect[i];
			intersect_var_sizes.push_back(var_sizes[var_in_intersect]);
			vector<int32_t>::iterator iter = find(m_region_inds.begin(), m_region_inds.end(), var_in_intersect);

			// Verify that the current intersection variable indeed appears in this region, and get the index where it appears
			if (iter == m_region_inds.end())
			{
				SG_SERROR("Intersection set contains variable %d which is not in region\n", var_in_intersect);
				return;
			}
			else
			{
				tmp_inds_of_intersects.push_back(iter - m_region_inds.begin());
			}
		}
		m_inds_of_intersects.push_back(tmp_inds_of_intersects);

		// This will initialize the message and  set it to zero
		CMDArray curr_msg(intersect_var_sizes);
		curr_msg = 0;

		m_msgs_from_region.push_back(curr_msg);
	}
	// Calculate the size of the region state space
	for (uint32_t i = 0; i < region_inds.size(); ++i)
	{
		m_var_sizes.push_back(var_sizes[region_inds[i]]);
	}
}

void CRegion::update_messages(vector<CMDArray> &sum_into_intersects)
{
	// First do the expansion:
	// 1. Take out the message into the intersection set from the current cluster
	// 2. Expand it to the size of the region
	// 3. Add this for all intersection sets
	// Set this to be the region's intersection set value
	CMDArray orig(sum_into_intersects[m_region_intersect]);
	for (uint32_t si = 0; si < m_intersect_inds.size(); ++si)
	{
		// Take out previous message
		vector<int32_t> &curr_inds_of_intersect = m_inds_of_intersects[si];
		CMDArray arr_expand(orig.m_base_sizes);
		m_msgs_from_region[si].expand(arr_expand, curr_inds_of_intersect);
		orig += arr_expand;
	}
	// Will store the total messages going into the intersection, but not from the Region
	vector<CMDArray> lam_minus_region;
	for (uint32_t si = 0; si < m_intersect_inds.size(); ++si)
	{
		int32_t curr_intersect = m_intersect_inds[si];
		lam_minus_region.push_back(CMDArray());
		lam_minus_region.back() = sum_into_intersects[curr_intersect];
		lam_minus_region.back() -= m_msgs_from_region[si];

		vector<int32_t> &curr_inds_of_intersect = m_inds_of_intersects[si];

		// If the intersection has the same size as the region, we assume they are the same, and therefore no need
		// to expand. NOTE: This may cause problems if the intersection has the same indices but rearranged.
		if ((uint32_t)get_num_vars() == curr_inds_of_intersect.size())
		{
			sum_into_intersects[m_region_intersect] += sum_into_intersects[curr_intersect];
		}
		else
		{
			CMDArray arr_expand(sum_into_intersects[m_region_intersect].m_base_sizes);
			sum_into_intersects[curr_intersect].expand(arr_expand, curr_inds_of_intersect);
			sum_into_intersects[m_region_intersect] += arr_expand;
		}
	}
	// Update messages
	max_into_multiple_subsets(sum_into_intersects[m_region_intersect], m_inds_of_intersects, m_msgs_from_region); // sets m_msgs_from_region
	int32_t sC = m_intersect_inds.size();
	for (uint32_t si = 0; si < m_intersect_inds.size(); ++si)
	{
		// Take out previous message
		int32_t curr_intersect = m_intersect_inds[si];
		// Update message
		m_msgs_from_region[si] *= 1.0 / sC;
		// Put in current message
		sum_into_intersects[curr_intersect] = m_msgs_from_region[si];
		// Finish updating message
		// msg_new = new - old + msg_old
		m_msgs_from_region[si] -= lam_minus_region[si];
		// Update region intersection set
		vector<int32_t> &curr_inds_of_intersect = m_inds_of_intersects[si];
		CMDArray arr_expand(orig.m_base_sizes);
		m_msgs_from_region[si].expand(arr_expand, curr_inds_of_intersect);
		orig -= arr_expand;

	}
	memcpy(sum_into_intersects[m_region_intersect].array, orig.array, orig.m_flat_length * sizeof(float64_t));

	return;
}

void CRegion::max_into_multiple_subsets(CMDArray big_arr, vector<vector<int32_t> > &all_subset_inds, vector<CMDArray> &all_max_res) const
{
	uint32_t nSubsets = all_subset_inds.size();
	bool* b_need_max = new bool[nSubsets];

	for (uint32_t si = 0; si < nSubsets; si++)
	{
		// If the subset equals the big array then maximizing would
		// give us the subset (assuming there is no reordering)
		if (all_subset_inds[si].size() == big_arr.m_base_sizes.size())
		{
			all_max_res[si] = big_arr;
			b_need_max[si] = 0;
		}
		else
		{
			all_max_res[si] = -1e9;
			b_need_max[si] = 1;
		}
	}

	// Go over all values of the big_array. For each check if its
	// value on the subset is larger than the current max
	vector<int32_t> inds_for_big(big_arr.m_base_sizes.size(), 0);

	for (int32_t vi = 0; vi < big_arr.m_flat_length; vi++)
	{
		for (uint32_t si = 0; si < nSubsets; si++)
		{
			if (!b_need_max[si])
			{
				continue;
			}
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
				y = inds_for_big[ind1] * all_max_res[si].m_base_sizes[1] + inds_for_big[ind2];
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
	SGVector<int32_t> cardinalities = m_fg->get_cardinalities();
	CDynamicObjectArray* fg_factors = m_fg->get_factors();
	int32_t num_factors = fg_factors->get_num_elements();

	std::vector< std::vector<int32_t> > all_factors(num_factors); // all_factors[i] stores the indeces of variables of factor i
	std::vector< std::vector<float64_t> > all_lambdas(num_factors); // the lambdas (messages) are log(energy table), all_lambdas[i] stores the energy table of factor i

	for (int32_t fi = 0; fi < num_factors; fi++)
	{
		CFactor* fac = dynamic_cast<CFactor*>(fg_factors->get_element(fi));
		SGVector<int32_t> vars = fac->get_variables();

		vector<int32_t> vec_vars(vars.size());

		for (int32_t vi = 0; vi < vars.size(); vi++)
		{
			vec_vars[vi] = vars[vi];
		}
		all_factors[fi] = vec_vars;

		SGVector<float64_t> energies = fac->get_energies();

		for (int32_t ei = 0; ei < energies.size(); ei++)
		{
			all_lambdas[fi].push_back(-energies[ei]);
		}

		if (all_lambdas[fi].size() == 4)
		{
			float64_t val = all_lambdas[fi][1];
			all_lambdas[fi][1] = all_lambdas[fi][2];
			all_lambdas[fi][2] = val;
		}

		SG_UNREF(fac);
	}

	SG_UNREF(fg_factors);

	std::vector<int32_t> var_sizes(cardinalities.size());

	for (uint32_t i = 0; i < var_sizes.size(); i++)
	{
		var_sizes[i] = cardinalities[i];
	}

	init(var_sizes, all_factors, all_lambdas);
}

void CMPLP::init(vector<int32_t> var_sizes, vector<vector<int32_t> > all_region_inds, vector<vector<float64_t> > all_lambdas)
{
	m_var_sizes = var_sizes;

	// Set the intersection sets to be all single nodes and also all regions
	// Initialize sum into intersections.

	// First, add all individual nodes as their own intersection set
	for (uint32_t si = 0; si < m_var_sizes.size(); ++si)
	{
		m_all_intersects.push_back(vector<int32_t>(1, si));
		vector<int32_t> subset_size(1, m_var_sizes[si]);   // Initialize sum into intersections to zero for these
		m_sum_into_intersects.push_back(CMDArray(subset_size) = 0);
		m_single_node_lambdas.push_back(CMDArray(subset_size) = 0);
	}

	// Next initialize all regions. If not a single node, give them their own intersection set
	for (uint32_t ri = 0; ri < all_region_inds.size(); ++ri)
	{
		vector<int32_t> region_var_sizes;
		for (uint32_t i = 0; i < all_region_inds[ri].size(); ++i)
		{
			region_var_sizes.push_back(m_var_sizes[all_region_inds[ri][i]]);
		}

		if (all_region_inds[ri].size() == 0)
		{
			SG_DEBUG("Length zero region!\n");
		}

		if (all_region_inds[ri].size() == 1 && all_lambdas[ri].size() != 0)
		{
			CMDArray curr_lambda(region_var_sizes);
			for (int32_t i = 0; i < curr_lambda.m_flat_length; ++i)
			{
				curr_lambda[i] = all_lambdas[ri][i];
			}
			// Insert the single node potential into sum_into_intersects
			m_sum_into_intersects[all_region_inds[ri][0]] += curr_lambda;
			m_single_node_lambdas[all_region_inds[ri][0]] += curr_lambda;
		}
		else
		{
			vector<int32_t> curr_intersect(all_region_inds[ri]);
			m_all_intersects.push_back(curr_intersect);
			int32_t curr_intersect_loc = m_all_intersects.size() - 1;
			if (all_lambdas[ri].size() != 0)
			{
				CMDArray curr_lambda = CMDArray(region_var_sizes);

				// Assume all_lambdas is given as a "flat" vector and put it into curr_lambda
				for (int32_t i = 0; i < curr_lambda.m_flat_length; ++i)
				{
					curr_lambda[i] = all_lambdas[ri][i];
				}
				m_sum_into_intersects.push_back(curr_lambda);

				CRegion curr_region(all_region_inds[ri], m_all_intersects, all_region_inds[ri], m_var_sizes, curr_intersect_loc);

				m_all_regions.push_back(curr_region);
				m_region_lambdas.push_back(CMDArray(curr_lambda));
			}
			else    // Empty constructor
			{
				CRegion curr_region(all_region_inds[ri], m_all_intersects, all_region_inds[ri], m_var_sizes, curr_intersect_loc);
				m_all_regions.push_back(curr_region);
				m_region_lambdas.push_back(CMDArray());
			}
			// If this is an edge, insert into the map
			if (curr_intersect.size() == 2)
			{
				// First sort
				vector<int32_t> tmp_inds(curr_intersect);
				sort(tmp_inds.begin(), tmp_inds.end());
				// Then insert
				m_intersect_map.insert(pair<pair<int, int>, int>(pair<int, int>(tmp_inds[0], tmp_inds[1]), curr_intersect_loc));
			}
		}
	}

	// Initialize output vector
	for (uint32_t i = 0; i < m_var_sizes.size(); ++i)
	{
		m_decoded_res.push_back(0);
		m_best_decoded_res.push_back(0);
	}
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
	{
		assignment[vi] = m_best_decoded_res[vi];
	}

	float64_t energy = m_fg->evaluate_energy(assignment);
	SG_DEBUG("fg.evaluate_energy(assignment) = %f\n", energy);

	return energy;
}

void CMPLP::run_mplp(int32_t niter, float64_t obj_del_thr, float64_t int_gap_thr)
{
	// Perform the GMPLP updates (Sontag's modified version), not quite as in the GJ NIPS07 paper
	for (int32_t it = 0; it < niter; ++it)
	{

		for (uint32_t ri = 0; ri < m_all_regions.size(); ++ri)
		{
			m_all_regions[ri].update_messages(m_sum_into_intersects);
		}

		float64_t obj;
		float64_t int_gap;

		obj = local_decode();
		m_obj_del = m_last_obj - obj;
		m_last_obj = obj;

		int_gap = obj - m_best_val;

		SG_SDEBUG("Iter= %d Objective=%f Decoded=%f ObjDel=%f IntGap=%f \n", (it + 1), obj, m_best_val, m_obj_del, int_gap);

		if (m_obj_del < obj_del_thr && it > m_param.m_max_iter_inner)
		{
			break;
		}
		if (int_gap < int_gap_thr)
		{
			break;
		}
	}
	return;
}

float64_t CMPLP::compute_int_val(vector<int32_t> &assignment) const
{
	float64_t int_val = 0;
	for (uint32_t ri = 0; ri < m_all_regions.size(); ++ri)
	{
		if (m_region_lambdas[ri].m_flat_length)
		{
			vector<int32_t> tmpvec;
			for (uint32_t vi = 0; vi < m_all_regions[ri].m_region_inds.size(); ++vi)
			{
				tmpvec.push_back(assignment[m_all_regions[ri].m_region_inds[vi]]);
			}
			int_val += m_region_lambdas[ri].get_value(tmpvec);
		}
	}
	//This iterates over all singletons
	for (uint32_t ni = 0; ni < m_var_sizes.size(); ++ni)
	{
		int_val += m_single_node_lambdas[ni][assignment[ni]];
	}
	return int_val;
}

float64_t CMPLP::local_decode()
{
	float64_t obj = 0;
	int32_t max_at;
	for (uint32_t si = 0; si < m_sum_into_intersects.size(); ++si)
	{
		obj += m_sum_into_intersects[si].max_element(max_at);
		// If this is a singleton, keep its value (so that we also have an integral assignment).
		// NOTE: Here we assume that all singletons are intersection sets. Otherwise, some variables will not be decoded here
		if (m_all_intersects[si].size() == 1)
		{
			m_decoded_res[m_all_intersects[si][0]] = max_at;
		}
	}
	update_result();
	return obj;
}

float64_t CMPLP::update_result()
{
	float64_t int_val;
	if ((int_val = compute_int_val(m_decoded_res)) > m_best_val)
	{
		SG_SDEBUG("int val: %f\n", int_val);
		m_best_decoded_res.assign(m_decoded_res.begin(), m_decoded_res.end());
		m_best_val = int_val;
	}
	return int_val;
}
