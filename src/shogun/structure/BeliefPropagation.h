/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu
 * Copyright (C) 2013 Shell Hu
 */

#ifndef __BELIEF_PROPAGATION_H__
#define __BELIEF_PROPAGATION_H__

#include <shogun/lib/config.h>

#include <shogun/lib/SGVector.h>
#include <shogun/structure/FactorGraph.h>
#include <shogun/structure/MAPInference.h>

#include <vector>
#include <set>

#include <unordered_map>

#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace shogun
{
#define IGNORE_IN_CLASSLIST

enum ENodeType
{
	VAR_NODE = 0,
	FAC_NODE = 1
};

enum EEdgeType
{
	VAR_TO_FAC = 0,
	FAC_TO_VAR = 1
};

struct GraphNode
{
	index_t node_id;
	ENodeType node_type; // 1 var, 0 factor
	index_t parent;      // where came from

	GraphNode(index_t id, ENodeType type, index_t pa)
		: node_id(id), node_type(type), parent(pa)
	{
	}
	~GraphNode() { }
};

struct MessageEdge
{
	EEdgeType mtype; // 1 var_to_factor, 0 factor_to_var
	index_t child;
	index_t parent;

	MessageEdge(EEdgeType type, index_t ch, index_t pa)
		: mtype(type), child(ch), parent(pa)
	{
	}

	~MessageEdge() { }

	inline index_t get_var_node()
	{
		return mtype == VAR_TO_FAC ? child : parent;
	}

	inline index_t get_factor_node()
	{
		return mtype == VAR_TO_FAC ? parent : child;
	}
};

/** If tree structure, do exact inference, otherwise loopy belief propagation */
IGNORE_IN_CLASSLIST class CBeliefPropagation : public CMAPInferImpl
{
public:
	CBeliefPropagation();
	CBeliefPropagation(CFactorGraph* fg);

	virtual ~CBeliefPropagation();

	/** @return class name */
	virtual const char* get_name() const { return "BeliefPropagation"; }

	virtual float64_t inference(SGVector<index_t> assignment);

protected:
	float64_t m_map_energy;
};

/** max-product algorithm for tree graph
 * please refer to algorithm 1 on page 44 of [1] for more detail.
 *
 * [1] Sebastian Nowozin and Christoph H. Lampert,
 * Structured Learning and Prediction for Computer Vision,
 * Foundations and Trends in Computer Graphics and Vision series
 * of now publishers, 2011.
 */
IGNORE_IN_CLASSLIST class CTreeMaxProduct : public CBeliefPropagation
{
	typedef std::unordered_map<uint32_t, uint32_t> msg_map_type;
	typedef std::unordered_map<uint32_t, std::set<uint32_t> > msgset_map_type;
	typedef std::unordered_multimap<index_t, index_t> var_factor_map_type;

public:
	CTreeMaxProduct();
	CTreeMaxProduct(CFactorGraph* fg);

	virtual ~CTreeMaxProduct();

	/** @return class name */
	virtual const char* get_name() const { return "TreeMaxProduct"; }

	virtual float64_t inference(SGVector<index_t> assignment);

protected:
	void bottom_up_pass();
	void top_down_pass();
	void get_message_order(std::vector<MessageEdge*>& order, std::vector<bool>& is_root) const;

private:
	void init();

private:
	std::vector<MessageEdge*> m_msg_order;
	std::vector<bool> m_is_root;
	std::vector< std::vector<float64_t> > m_fw_msgs;
	std::vector< std::vector<float64_t> > m_bw_msgs;
	std::vector<index_t> m_states;

	msg_map_type m_msg_map_var;
	msg_map_type m_msg_map_fac;
	msgset_map_type m_msgset_map_var;
};

}

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

#endif
