/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Shell Hu, Yuyu Zhang, Bjoern Esser
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
IGNORE_IN_CLASSLIST class BeliefPropagation : public MAPInferImpl
{
public:
	BeliefPropagation();
	BeliefPropagation(std::shared_ptr<FactorGraph> fg);

	~BeliefPropagation() override;

	/** @return class name */
	const char* get_name() const override { return "BeliefPropagation"; }

	float64_t inference(SGVector<int32_t> assignment) override;

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
IGNORE_IN_CLASSLIST class TreeMaxProduct : public BeliefPropagation
{
	typedef std::unordered_map<uint32_t, uint32_t> msg_map_type;
	typedef std::unordered_map<uint32_t, std::set<uint32_t> > msgset_map_type;
	typedef std::unordered_multimap<int32_t, int32_t> var_factor_map_type;

public:
	TreeMaxProduct();
	TreeMaxProduct(std::shared_ptr<FactorGraph> fg);

	~TreeMaxProduct() override;

	/** @return class name */
	const char* get_name() const override { return "TreeMaxProduct"; }

	float64_t inference(SGVector<int32_t> assignment) override;

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
