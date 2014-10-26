/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Jiaolong Xu
 * Copyright (C) 2014 Jiaolong Xu
 */

#ifndef __GRAPH_CUT_H__
#define __GRAPH_CUT_H__

#include <shogun/lib/config.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/List.h>
#include <shogun/base/DynArray.h>
#include <shogun/structure/FactorGraph.h>
#include <shogun/structure/Factor.h>
#include <shogun/structure/MAPInference.h>

/* special constants for node->parent. */
#define TERMINAL_EDGE ( (GCEdge *) 1 ) // to terminal
#define ORPHAN_EDGE   ( (GCEdge *) 2 ) // orphan

#define INFINITE_D 1000000000 // infinite distance to the terminal

namespace shogun
{
/** Graph cuts terminal type */
enum ETerminalType
{
	/** source terminal */
	SOURCE = 0,
	/** sink terminal */
	SINK = 1 
};

struct GCNode;
/** @brief Graph cuts edge
 *
 */
struct GCEdge
{
	/** edge id */
	int32_t id;
	/** node the edge point to */
	GCNode* head;
	/** next edge with the same originated node */
	GCEdge* next;
	/** reverse edge */
	GCEdge* reverse;
	/** residual capacity */
	float64_t residual_capacity;
};

/** @brief Graph cuts node
 *
 */
struct GCNode
{
	/** node id */
	int32_t id;
	/** first outcoming edge */
	GCEdge* first;
	/** node's parent */
	GCEdge* parent;
	/** pointer to the next active node
	 * (or itself if it is the last node in the list)
	 */
	GCNode* next;
	/** timestamp showing when dist_to_terminal was computed */
	int32_t timestamp;
	/** distance to the terminal */
	int32_t dist_terminal;
	/** the type of the tree that the node belongs to */
	ETerminalType type_tree;
	/** if tree_cap > 0 then tree_cap is residual capacity
	 * of the edge SOURCE->node otherwise -tree_cap is 
	 * residual capacity of the edge node->SINK */
	float64_t tree_cap;
};

/** @brief Graph guts node pointer
 *
 */
struct GCNodePtr
{
	/** pointer of the node */
	GCNode* ptr;
	/** point to the next */
	GCNodePtr* next;
};

/** Graph cuts inference for fatcor graph using V. Kolmogorov's max-flow/min-cut algorithm
 *
 * Please refer to the paper for more details:
 *
 * "An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision."
 * Yuri Boykov and Vladimir Kolmogorov.
 * In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2004.
 *
 * Currently, only binary lablel is supported, factor order <= 3
 */
class CGraphCut : public CMAPInferImpl
{
public:
	/** Constructor */
	CGraphCut();

	/** Constructor
	 *
	 * @param fg factor graph
	 */
	CGraphCut(CFactorGraph* fg);

	/** Constructor
	 * This constructor is used for general s-t graph, the next steps to compute the max flow are:
	 * 1. use add_tweight and add_edges to add nodes and edges to the s-t graph
	 * 2. init_maxflow()
	 * 3. compute_maxflow()
	 *
	 * @param num_nodes number of nodes in the s-t graph (SOURCE and SINK nodes are not included)
	 * @param num_edges number of edges in the s-t graph (the edges connecting to the SOURCE and SINK are not included)
	 */
	CGraphCut(int32_t num_nodes, int32_t num_edges);

	/** Destructor */
	virtual ~CGraphCut();

	/** @return class name */
	virtual const char* get_name() const
	{
		return "GraphCut";
	}

	/** Inference
	 *
	 * @param assignment the assignment
	 * @return the total energy after doing inference
	 */
	virtual float64_t inference(SGVector<int32_t> assignment);

	/** @return the total flow in the s-t graph */
	float64_t get_flow()
	{
		return m_flow;
	}

	/** Build s-t graph 
	 *
	 * @param num_nodes number of nodes in the s-t graph (SOURCE and SINK nodes are not included)
	 * @param num_edges number of edges in the s-t graph (the edges connecting to the SOURCE and SINK are not included)
	 */
	void build_st_graph(int32_t num_nodes, int32_t num_edges);

	/** Adds a bidirectional edge between 'i' and 'j' with the weights 'cap' and 'rev_cap'.
	 *
	 * @param i node id i
	 * @param j node id j
	 * @param capacity edge capacity
	 * @param reverse_capacity reverse edge capacity
	 */
	void add_edge(int32_t i, int32_t j, float64_t capacity, float64_t reverse_capacity);

	/** Adds new edges 'SOURCE->i' and 'i->SINK' with corresponding weights.
	 * Can be called multiple times for each node. Weights can be negative.
	 * NOTE: the number of such edges is not counted in edge_num_max.
	 * No internal memory is allocated by this call.
	 *
	 * @param i node id i
	 * @param cap_source SOURCE->i capacity
	 * @param cap_sink i->SINK capacity
	 */
	void add_tweights(int32_t i, float64_t cap_source, float64_t cap_sink);

	/** Initialize max flow, call this function after adding nodes and edges */
	void init_maxflow();

	/** Compute the maxflow
	 *
	 * The algorithm builds up two search trees, a source-tree and a sink-tree.
	 * In the beginning of the algorithm only the source and the sink nodes and all other
	 * nodes are free nodes. The algorithm consists of three phases:
	 *
	 * grow-phase: In this phase active nodes aquire new children from a set of free nodes.
	 * The grow phase terminates if there are no more active nodes left or a node discovers
	 * a node from the other search tree through an unsaturated edge. In this case a path
	 * from source to sink is found.
	 *
	 * augmentation-pahse: the input for this stage is a path P from s to t found at the growth
	 * stage. Some nodes may become "orphans" because their parents are no longer valid as they
	 * are on the augmented path and saturated.
	 *
	 * adoption-phase: the adoption stage tries to find a new valid parent for each orphan. A parent
	 * should be in the same set (SOURCE or SINK), as the orphan and connected through a non-saturated
	 * edge. The stage terminates when no orphans are left and the search tree of SOURCE and SINK are
	 * restored.
	 *
	 * After the adoption stage is completed, the algorithm returns to the growth stage. The algorithm
	 * terminates when the search tree cannot grow (no active nodes) and the trees are seperated by
	 * saturated edges. This implies that a maximum flow is achieved.
	 *
	 * @return the total flow after computing max flow
	 */
	float64_t compute_maxflow();

	/** After the maxflow is computed, this function returns to which
	 * terminal the node 'i' belongs (ETerminalType::SOURCE or ETerminalType::SINK).
	 *
	 * Occasionally there may be several minimum cuts. If a node can be assigned
	 * to both the source and the sink, then default_terminal is returned.
	 *
	 * @param i node id
	 * @param default_terminal default terminal
	 *
	 * @return terminal that the node belongs to
	 */
	ETerminalType get_assignment(int32_t i, ETerminalType default_terminal = SOURCE);

	/** Print the s-t graph */
	void print_graph();

	/** Print assignment */
	void print_assignment();

private:
	/** Initialize graph cuts */
	void init();

	/** Add a factor to the s-t graph
	 * More details about this function can be found in section 4 and section 5 in
	 * V. Kolmogorov, and R. Zabin. "What energy functions can be minimized via graph cuts?."
	 * T-PAMI. 2004.
	 *
	 * @param factor the factor to add
	 */
	void add_factor(CFactor* factor);

	/** Get the triple node id in s-t graph (for factor order = 3)
	 *
	 * @param triple triple is a vector contains the id of three nodes
	 * @return the id of triple node in the s-t graph
	 */
	int32_t get_tripleId(SGVector<int32_t> triple);

	/** Add a node to the active list
	 *
	 * node_i->next points to the next node in the list
	 * (or to node_i, if node_i is the last node in the list).
	 * If node_i->next is NULL iff node_i is not in the list.
	 *
	 * There are two queues. Active nodes are added to the end
	 * of the second queue and read from the front of the first
	 * queue. If the first queue is empty, it is replaced by the
	 * second queue(and the second queue becomes empty).
	 *
	 * @param node_i the node to add
	 */
	void set_active(GCNode* node_i);

	/** Get an active node next to the current node from the list */
	GCNode* next_active();

	/** Add node to the beginning of the orphan list
	 *
	 * @param node_i the node to add
	 */
	void set_orphan_front(GCNode* node_i);

	/** Add node to the end of the orphan list
	 *
	 * @param node_i the node to add
	 */
	void set_orphan_rear(GCNode* node_i);

	/** Process an orphan node
	 *
	 * @param node_i the node to process
	 * @param terminalType_tree SOURCE or SINK tree
	 */
	void process_orphan(GCNode* node_i, ETerminalType terminalType_tree);

	/** Grow to add node to active set
	 *
	 * @param edge connecting edge on a source->sink path
	 * @param current_node current node
	 *
	 * @return true if the next active node is found otherwise false
	 */
	bool grow(GCEdge* &edge, GCNode* &current_node);

	/** Augment the source->sink path
	 *
	 * @param connecting_edge the edge connecting a source->sink path
	 */
	void augment_path(GCEdge* connecting_edge);

	/** Adopt orphan nodes */
	void adopt();

	/** Test the consistency of the graph, for debug mode */
	void test_consistency(GCNode* current_node = NULL);
protected:
	/** the total energy of the factor graph */
	float64_t m_map_energy;

private:
	/** number of variables in the factor graph */
	int32_t m_num_variables;
	/** statistic of the number of the factors at order [1, 2, 3] */
	SGVector<int32_t> m_num_factors_at_order;
	/** list of triple nodes, for order-3 factors */
	DynArray< SGVector<int32_t> > m_triple_list;

	/** nodes in the st graph */
	GCNode*		m_nodes;
	/** number of nodes */
	int32_t		m_num_nodes;
	/** edges in the st graph */
	GCEdge*		m_edges;
	/** point to the end of the added edges */
	GCEdge*		m_edges_last;
	/** number of edges */
	int32_t		m_num_edges;

	/** total flow */
	float64_t	m_flow;
	/** timestamp */
	int32_t		m_timestamp;

	/** list of active nodes */
	GCNode*		m_active_first[2];
	/** list of active nodes */
	GCNode*		m_active_last[2];

	/** list of pointers to orphans */
	GCNodePtr*	m_orphan_first;
	/** list of pointers to orphans */
	GCNodePtr*	m_orphan_last;
};

}
#endif
