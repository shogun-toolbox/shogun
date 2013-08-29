/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu 
 * Copyright (C) 2013 Shell Hu 
 */

#include <shogun/structure/BeliefPropagation.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/io/SGIO.h>
#include <numeric>
#include <algorithm>
#include <functional>
#include <stack>

using namespace shogun;

CBeliefPropagation::CBeliefPropagation()
	: CMAPInferImpl()
{
	SG_UNSTABLE("CBeliefPropagation::CBeliefPropagation()", "\n");
}

CBeliefPropagation::CBeliefPropagation(CFactorGraph* fg)
	: CMAPInferImpl(fg)
{
}

CBeliefPropagation::~CBeliefPropagation()
{
}

float64_t CBeliefPropagation::inference(SGVector<int32_t> assignment)
{
	SG_ERROR("%s::inference(): please use TreeMaxProduct or LoopyMaxProduct!\n", get_name());
	return 0;
}

// -----------------------------------------------------------------

CTreeMaxProduct::CTreeMaxProduct()
	: CBeliefPropagation()
{
	SG_UNSTABLE("CTreeMaxProduct::CTreeMaxProduct()", "\n");

	init();
}

CTreeMaxProduct::CTreeMaxProduct(CFactorGraph* fg)
	: CBeliefPropagation(fg)
{
	ASSERT(m_fg != NULL);

	init();

	CDisjointSet* dset = m_fg->get_disjoint_set();
	bool is_connected = dset->get_connected();
	SG_UNREF(dset);

	if (!is_connected)
		m_fg->connect_components();

	get_message_order(m_msg_order, m_is_root);

	// calculate lookup tables for forward messages
	// a key is unique because a tree has only one root
	// a var or a factor has only one edge towards root
	for (uint32_t mi = 0; mi < m_msg_order.size(); mi++)
	{
		if (m_msg_order[mi]->mtype == VAR_TO_FAC) // var_to_factor
		{
			// <var_id, msg_id>
			m_msg_map_var[m_msg_order[mi]->child] = mi;
		}
		else // factor_to_var
		{
			// <fac_id, msg_id>
			m_msg_map_fac[m_msg_order[mi]->child] = mi;
			// collect incoming msgs for each var_id
			m_msgset_map_var[m_msg_order[mi]->parent].insert(mi);
		}
	}

}

CTreeMaxProduct::~CTreeMaxProduct()
{
	if (!m_msg_order.empty())
	{
		for (std::vector<MessageEdge*>::iterator it = m_msg_order.begin(); it != m_msg_order.end(); ++it)
			delete *it;
	}
}

void CTreeMaxProduct::init()
{
	m_msg_order = std::vector<MessageEdge*>(m_fg->get_num_edges(), (MessageEdge*) NULL);
	m_is_root = std::vector<bool>(m_fg->get_cardinalities().size(), false);
	m_fw_msgs = std::vector< std::vector<float64_t> >(m_msg_order.size(), 
			std::vector<float64_t>());
	m_bw_msgs = std::vector< std::vector<float64_t> >(m_msg_order.size(), 
			std::vector<float64_t>());
	m_states = std::vector<int32_t>(m_fg->get_cardinalities().size(), 0);

	m_msg_map_var = msg_map_type(); 
	m_msg_map_fac = msg_map_type(); 
	m_msgset_map_var = msgset_map_type(); 
}

void CTreeMaxProduct::get_message_order(std::vector<MessageEdge*>& order, 
	std::vector<bool>& is_root) const
{
	ASSERT(m_fg->is_acyclic_graph());

	// 1) pick up roots according to union process of disjoint sets 
	CDisjointSet* dset = m_fg->get_disjoint_set();
	if (!dset->get_connected())
	{
		SG_UNREF(dset);
		SG_ERROR("%s::get_root_indicators(): run connect_components() first!\n", get_name());
	}

	int32_t num_vars = m_fg->get_cardinalities().size();
	if (is_root.size() != (uint32_t)num_vars)
		is_root.resize(num_vars);

	std::fill(is_root.begin(), is_root.end(), false);
	
	for (int32_t vi = 0; vi < num_vars; vi++)
		is_root[dset->find_set(vi)] = true;

	SG_UNREF(dset);
	ASSERT(std::accumulate(is_root.begin(), is_root.end(), 0) >= 1);

	// 2) caculate message order
	// <var_id, fac_id>
	var_factor_map_type vf_map;
	CDynamicObjectArray* facs = m_fg->get_factors();

	for (int32_t fi = 0; fi < facs->get_num_elements(); ++fi)
	{
		CFactor* fac = dynamic_cast<CFactor*>(facs->get_element(fi));
		SGVector<int32_t> vars = fac->get_variables();
		for (int32_t vi = 0; vi < vars.size(); vi++)
			vf_map.insert(var_factor_map_type::value_type(vars[vi], fi));

		SG_UNREF(fac);
	}

	std::stack<GraphNode*> node_stack;
	// init node_stack with root nodes
	for (uint32_t ni = 0; ni < is_root.size(); ni++)
	{
		if (is_root[ni])
		{
			// node_id = ni, node_type = variable, parent = none
			node_stack.push(new GraphNode(ni, VAR_NODE, -1));
		}
	}

	if (order.size() != (uint32_t)(m_fg->get_num_edges()))
		order.resize(m_fg->get_num_edges());

	// find reverse order
	int32_t eid = m_fg->get_num_edges() - 1;
	while (!node_stack.empty())
	{
		GraphNode* node = node_stack.top();
		node_stack.pop();

		if (node->node_type == VAR_NODE) // child: factor -> parent: var
		{
			typedef var_factor_map_type::const_iterator const_iter;
			std::pair<const_iter, const_iter> adj_factors = vf_map.equal_range(node->node_id);
			for (const_iter mi = adj_factors.first; mi != adj_factors.second; ++mi) 
			{
				int32_t adj_factor_id = mi->second;
				if (adj_factor_id == node->parent)
					continue;

				order[eid--] = new MessageEdge(FAC_TO_VAR, adj_factor_id, node->node_id);
				// add factor node to node_stack
				node_stack.push(new GraphNode(adj_factor_id, FAC_NODE, node->node_id));
			}
		}
		else // child: var -> parent: factor
		{
			CFactor* fac = dynamic_cast<CFactor*>(facs->get_element(node->node_id));
			SGVector<int32_t> vars = fac->get_variables();
			SG_UNREF(fac);

			for (int32_t vi = 0; vi < vars.size(); vi++)
			{
				if (vars[vi] == node->parent)
					continue;

				order[eid--] = new MessageEdge(VAR_TO_FAC, vars[vi], node->node_id);
				// add variable node to node_stack
				node_stack.push(new GraphNode(vars[vi], VAR_NODE, node->node_id));
			}
		}

		delete node;
	}

	SG_UNREF(facs);
}

float64_t CTreeMaxProduct::inference(SGVector<int32_t> assignment)
{
	REQUIRE(assignment.size() == m_fg->get_cardinalities().size(),
		"%s::inference(): the output assignment should be prepared as"
		"the same size as variables!\n", get_name());

	bottom_up_pass();
	top_down_pass();

	for (int32_t vi = 0; vi < assignment.size(); vi++)
		assignment[vi] = m_states[vi];
	
	SG_DEBUG("fg.evaluate_energy(assignment) = %f\n", m_fg->evaluate_energy(assignment));
	SG_DEBUG("minimized energy = %f\n", -m_map_energy);

	return -m_map_energy;
}

void CTreeMaxProduct::bottom_up_pass()
{
	SG_DEBUG("\n***enter bottom_up_pass().\n");
	CDynamicObjectArray* facs = m_fg->get_factors();
	SGVector<int32_t> cards = m_fg->get_cardinalities();

	// init forward msgs to 0
	m_fw_msgs.resize(m_msg_order.size());
	for (uint32_t mi = 0; mi < m_msg_order.size(); ++mi)
	{
		// msg size is determined by var node of msg edge
		m_fw_msgs[mi].resize(cards[m_msg_order[mi]->get_var_node()]);
		std::fill(m_fw_msgs[mi].begin(), m_fw_msgs[mi].end(), 0);
	}

	// pass msgs along the order up to root
	// if var -> factor
	//   compute q_v2f
	// else factor -> var
	//   compute r_f2v
	// where q_v2f and r_f2v are beliefs of the edge collecting from neighborhoods
	// by one end, which will be sent to another end, read Eq.(3.19), Eq.(3.20) 
	// on [Nowozin et al. 2011] for more detail.
	for (uint32_t mi = 0; mi < m_msg_order.size(); ++mi)
	{
		SG_DEBUG("mi = %d, mtype: %d %d -> %d\n", mi, 
			m_msg_order[mi]->mtype, m_msg_order[mi]->child, m_msg_order[mi]->parent);
		if (m_msg_order[mi]->mtype == VAR_TO_FAC) // var -> factor
		{
			uint32_t var_id = m_msg_order[mi]->child;
			const std::set<uint32_t>& msgset_var = m_msgset_map_var[var_id];

			// q_v2f = sum(r_f2v), i.e. sum all incoming f2v msgs
			for (std::set<uint32_t>::const_iterator cit = msgset_var.begin(); cit != msgset_var.end(); cit++)
			{
				std::transform(m_fw_msgs[*cit].begin(), m_fw_msgs[*cit].end(),
					m_fw_msgs[mi].begin(), 
					m_fw_msgs[mi].begin(),
					std::plus<float64_t>());
			}
		}
		else // factor -> var
		{
			int32_t fac_id = m_msg_order[mi]->child;
			int32_t var_id = m_msg_order[mi]->parent;

			CFactor* fac = dynamic_cast<CFactor*>(facs->get_element(fac_id));
			CTableFactorType* ftype = fac->get_factor_type();
			SGVector<int32_t> fvars = fac->get_variables();
			SGVector<float64_t> fenrgs = fac->get_energies();
			SG_UNREF(fac);

			// find index of var_id in the factor
			SGVector<int32_t> fvar_set = fvars.find(var_id);
			ASSERT(fvar_set.vlen == 1);
			int32_t var_id_index = fvar_set[0]; 

			std::vector<float64_t> r_f2v(fenrgs.size(), 0);
			std::vector<float64_t> r_f2v_max(cards[var_id], 
				-std::numeric_limits<float64_t>::infinity());

			// TODO: optimize with index_from_new_state()
			// marginalization
			// r_f2v = max(-fenrg + sum_{j!=var_id} q_v2f[adj_var_state])
			for (int32_t ei = 0; ei < fenrgs.size(); ei++)
			{
				r_f2v[ei] = -fenrgs[ei];

				for (int32_t vi = 0; vi < fvars.size(); vi++)
				{
					if (vi == var_id_index)
						continue;

					uint32_t adj_msg = m_msg_map_var[fvars[vi]];
					int32_t adj_var_state = ftype->state_from_index(ei, vi);

					r_f2v[ei] += m_fw_msgs[adj_msg][adj_var_state];
				}

				// find max value for each state of var_id
				int32_t var_state = ftype->state_from_index(ei, var_id_index);
				if (r_f2v[ei] > r_f2v_max[var_state])
					r_f2v_max[var_state] = r_f2v[ei];
			}

			// in max-product, final r_f2v = r_f2v_max
			for (int32_t si = 0; si < cards[var_id]; si++)
				m_fw_msgs[mi][si] = r_f2v_max[si];

			SG_UNREF(ftype);
		}
	}
	SG_UNREF(facs);

	// -energy = max(sum_{f} r_f2root)
	m_map_energy = 0;
	for (uint32_t ri = 0; ri < m_is_root.size(); ri++)
	{
		if (!m_is_root[ri])
			continue;

		const std::set<uint32_t>& msgset_rt = m_msgset_map_var[ri];
		std::vector<float64_t> rmarg(cards[ri], 0);
		for (std::set<uint32_t>::const_iterator cit = msgset_rt.begin(); cit != msgset_rt.end(); cit++)
		{
			std::transform(m_fw_msgs[*cit].begin(), m_fw_msgs[*cit].end(),
				rmarg.begin(), 
				rmarg.begin(),
				std::plus<float64_t>());
		}

		m_map_energy += *std::max_element(rmarg.begin(), rmarg.end());
	}
	SG_DEBUG("***leave bottom_up_pass().\n");
}

void CTreeMaxProduct::top_down_pass()
{
	SG_DEBUG("\n***enter top_down_pass().\n");
	int32_t minf = std::numeric_limits<int32_t>::max();
	CDynamicObjectArray* facs = m_fg->get_factors();
	SGVector<int32_t> cards = m_fg->get_cardinalities();

	// init backward msgs to 0
	m_bw_msgs.resize(m_msg_order.size());
	for (uint32_t mi = 0; mi < m_msg_order.size(); ++mi)
	{
		// msg size is determined by var node of msg edge
		m_bw_msgs[mi].resize(cards[m_msg_order[mi]->get_var_node()]);
		std::fill(m_bw_msgs[mi].begin(), m_bw_msgs[mi].end(), 0);
	}

	// init states to max infinity
	m_states.resize(cards.size());
	std::fill(m_states.begin(), m_states.end(), minf);

	// infer states of roots first since marginal distributions of
	// root variables are ready after bottom-up pass
	for (uint32_t ri = 0; ri < m_is_root.size(); ri++)
	{
		if (!m_is_root[ri])
			continue;

		const std::set<uint32_t>& msgset_rt = m_msgset_map_var[ri];
		std::vector<float64_t> rmarg(cards[ri], 0);
		for (std::set<uint32_t>::const_iterator cit = msgset_rt.begin(); cit != msgset_rt.end(); cit++)
		{
			// rmarg += m_fw_msgs[*cit]
			std::transform(m_fw_msgs[*cit].begin(), m_fw_msgs[*cit].end(),
				rmarg.begin(), 
				rmarg.begin(),
				std::plus<float64_t>());
		}

		// argmax
		m_states[ri] = static_cast<int32_t>(
			std::max_element(rmarg.begin(), rmarg.end())
			- rmarg.begin());
	}

	// pass msgs down to leaf
	// if factor <- var edge
	//   compute q_v2f
	//   compute marginal of f
	// else var <- factor edge
	//   compute r_f2v
	for (int32_t mi = (int32_t)(m_msg_order.size()-1); mi >= 0; --mi)
	{
		SG_DEBUG("mi = %d, mtype: %d %d <- %d\n", mi, 
			m_msg_order[mi]->mtype, m_msg_order[mi]->child, m_msg_order[mi]->parent);
		if (m_msg_order[mi]->mtype == FAC_TO_VAR) // factor <- var
		{
			int32_t fac_id = m_msg_order[mi]->child;
			int32_t var_id = m_msg_order[mi]->parent;

			CFactor* fac = dynamic_cast<CFactor*>(facs->get_element(fac_id));
			CTableFactorType* ftype = fac->get_factor_type();
			SGVector<int32_t> fvars = fac->get_variables();
			SGVector<float64_t> fenrgs = fac->get_energies();
			SG_UNREF(fac);

			// find index of var_id in the factor
			SGVector<int32_t> fvar_set = fvars.find(var_id);
			ASSERT(fvar_set.vlen == 1);
			int32_t var_id_index = fvar_set[0]; 

			// q_v2f = r_bw_parent2v + sum_{child!=f} r_fw_child2v
			// make sure the state of var_id has been inferred (factor marginalization)
			// s.t. this msg computation will condition on the known state
			ASSERT(m_states[var_id] != minf);

			// parent msg: r_bw_parent2v
			if (m_is_root[var_id] == 0)
			{
				uint32_t parent_msg = m_msg_map_var[var_id];
				std::fill(m_bw_msgs[mi].begin(), m_bw_msgs[mi].end(), 
					m_bw_msgs[parent_msg][m_states[var_id]]);
			}

			// siblings: sum_{child!=f} r_fw_child2v
			const std::set<uint32_t>& msgset_var = m_msgset_map_var[var_id];
			for (std::set<uint32_t>::const_iterator cit = msgset_var.begin();
				cit != msgset_var.end(); cit++)
			{
				if (m_msg_order[*cit]->child == fac_id)
					continue;

				for (uint32_t xi = 0; xi < m_bw_msgs[mi].size(); xi++)
					m_bw_msgs[mi][xi] += m_fw_msgs[*cit][m_states[var_id]];
			}

			// m_states from maximizing marginal distributions of fac_id
			// mu(f) = -E(v_state) + sum_v q_v2f
			std::vector<float64_t> marg(fenrgs.size(), 0);
			for (uint32_t ei = 0; ei < marg.size(); ei++)
			{
				int32_t nei = ftype->index_from_new_state(ei, var_id_index, m_states[var_id]);
				marg[ei] = -fenrgs[nei];

				for (int32_t vi = 0; vi < fvars.size(); vi++)
				{
					if (vi == var_id_index)
					{
						int32_t var_id_state = ftype->state_from_index(ei, var_id_index);
						if (m_states[var_id] != minf)
							var_id_state = m_states[var_id];
						
						marg[ei] += m_bw_msgs[mi][var_id_state];
					}
					else
					{
						uint32_t adj_id = fvars[vi];
						uint32_t adj_msg = m_msg_map_var[adj_id];
						int32_t adj_id_state = ftype->state_from_index(ei, vi);

						marg[ei] += m_fw_msgs[adj_msg][adj_id_state];
					}
				}
			}

			int32_t ei_max = static_cast<int32_t>(
				std::max_element(marg.begin(), marg.end())
				- marg.begin());

			// infer states of neiboring vars of f
			for (int32_t vi = 0; vi < fvars.size(); vi++)
			{
				int32_t nvar_id = fvars[vi];
				// usually parent node has been inferred
				if (m_states[nvar_id] != minf)
					continue;

				int32_t nvar_id_state = ftype->state_from_index(ei_max, vi);
				m_states[nvar_id] = nvar_id_state;
			}

			SG_UNREF(ftype);
		}
		else // var <- factor
		{
			int32_t var_id = m_msg_order[mi]->child;
			int32_t fac_id = m_msg_order[mi]->parent;

			CFactor* fac = dynamic_cast<CFactor*>(facs->get_element(fac_id));
			CTableFactorType* ftype = fac->get_factor_type();
			SGVector<int32_t> fvars = fac->get_variables();
			SGVector<float64_t> fenrgs = fac->get_energies();
			SG_UNREF(fac);

			// find index of var_id in the factor
			SGVector<int32_t> fvar_set = fvars.find(var_id);
			ASSERT(fvar_set.vlen == 1);
			int32_t var_id_index = fvar_set[0]; 

			uint32_t msg_parent = m_msg_map_fac[fac_id];
			int32_t var_parent = m_msg_order[msg_parent]->parent;

			std::vector<float64_t> r_f2v(fenrgs.size());
			std::vector<float64_t> r_f2v_max(cards[var_id], 
				-std::numeric_limits<float64_t>::infinity());

			// r_f2v = max(-fenrg + sum_{j!=var_id} q_v2f[adj_var_state])
			for (int32_t ei = 0; ei < fenrgs.size(); ei++)
			{
				r_f2v[ei] = -fenrgs[ei];

				for (int32_t vi = 0; vi < fvars.size(); vi++)
				{
					if (vi == var_id_index)
						continue;

					if (fvars[vi] == var_parent)
					{
						ASSERT(m_states[var_parent] != minf);
						r_f2v[ei] += m_bw_msgs[msg_parent][m_states[var_parent]];
					}
					else
					{
						int32_t adj_id = fvars[vi];
						uint32_t adj_msg = m_msg_map_var[adj_id];
						int32_t adj_var_state = ftype->state_from_index(ei, vi);

						if (m_states[adj_id] != minf)
							adj_var_state = m_states[adj_id];

						r_f2v[ei] += m_fw_msgs[adj_msg][adj_var_state];
					}
				}

				// max marginalization
				int32_t var_id_state = ftype->state_from_index(ei, var_id_index);
				if (r_f2v[ei] > r_f2v_max[var_id_state])
					r_f2v_max[var_id_state] = r_f2v[ei];
			}

			for (int32_t si = 0; si < cards[var_id]; si++)
				m_bw_msgs[mi][si] = r_f2v_max[si];

			SG_UNREF(ftype);
		}
	} // end for msg edge

	SG_UNREF(facs);
	SG_DEBUG("***leave top_down_pass().\n");
}

