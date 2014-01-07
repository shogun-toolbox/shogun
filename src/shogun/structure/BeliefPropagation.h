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

#include <lib/SGVector.h>
#include <structure/FactorGraph.h>
#include <structure/MAPInference.h>

#include <vector>
#include <set>

#ifdef HAVE_STD_UNORDERED_MAP
	#include <unordered_map>
#else
	#include <tr1/unordered_map>
#endif

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
	int32_t node_id;
	ENodeType node_type; // 1 var, 0 factor
	int32_t parent; // where came from

	GraphNode(int32_t id, ENodeType type, int32_t pa)
		: node_id(id), node_type(type), parent(pa) { }
	~GraphNode() { }
};

struct MessageEdge
{
	EEdgeType mtype; // 1 var_to_factor, 0 factor_to_var
	int32_t child;
	int32_t parent;

	MessageEdge(EEdgeType type, int32_t ch, int32_t pa)
		: mtype(type), child(ch), parent(pa) { }

	~MessageEdge() { }

	inline int32_t get_var_node()
	{
		return mtype == VAR_TO_FAC ? child : parent;
	}

	inline int32_t get_factor_node()
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

	virtual float64_t inference(SGVector<int32_t> assignment);

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
#ifdef HAVE_STD_UNORDERED_MAP
	typedef std::unordered_map<uint32_t, uint32_t> msg_map_type;
	typedef std::unordered_map<uint32_t, std::set<uint32_t> > msgset_map_type;
	typedef std::unordered_multimap<int32_t, int32_t> var_factor_map_type;
#else
	typedef std::tr1::unordered_map<uint32_t, uint32_t> msg_map_type;
	typedef std::tr1::unordered_map<uint32_t, std::set<uint32_t> > msgset_map_type;
	typedef std::tr1::unordered_multimap<int32_t, int32_t> var_factor_map_type;
#endif

public:
	CTreeMaxProduct();
	CTreeMaxProduct(CFactorGraph* fg);

	virtual ~CTreeMaxProduct();

	/** @return class name */
	virtual const char* get_name() const { return "TreeMaxProduct"; }

	virtual float64_t inference(SGVector<int32_t> assignment);

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
	std::vector<int32_t> m_states;

	msg_map_type m_msg_map_var;
	msg_map_type m_msg_map_fac;
	msgset_map_type m_msgset_map_var;
};

}

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

#endif
