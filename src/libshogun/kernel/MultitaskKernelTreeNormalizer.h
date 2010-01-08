/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Christian Widmer
 * Copyright (C) 2010 Max-Planck-Society
 */

#ifndef _MULTITASKKERNELTREENORMALIZER_H___
#define _MULTITASKKERNELTREENORMALIZER_H___

#include "kernel/KernelNormalizer.h"
#include "kernel/Kernel.h"
#include <algorithm>
#include <map>
#include <set>

namespace shogun
{


class CNode: public CSGObject
{

public:


	float64_t beta;

	CNode() {
		parent = NULL;
		beta = 1.0;
	}

    /**
    fetch all ancesters of current node (excluding self)
    until root is reached including root

    @return: list of nodes on the path to root
    @rtype: list<TreeNode>
    **/
    std::set<CNode*> get_path_root() {

        std::set<CNode*> nodes_on_path = std::set<CNode*>();

        CNode* node = this;

        while (node != NULL) {
            nodes_on_path.insert(node);
            node = node->parent;
        }

        return nodes_on_path;
    }



    /**
    add node as child of current leaf

    @param node: child node
    @type node: TreeNode
    **/
	void add_child(CNode* node)
	{
        node->parent = this;
        this->children.push_back(node);
	}


	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "CNode";
	}


protected:

	CNode* parent;
	std::vector<CNode*> children;

};


class CTaxonomy : public CSGObject
{

public:


	CTaxonomy(){
		root = new CNode();
		nodes["root"] = root;
		names.push_back("root");
	}


	CNode* get_node(std::string task_id) {
		return nodes[task_id];
	}


	CNode* add_node(std::string parent_name, std::string child_name, float64_t beta) {

		CNode* child_node = new CNode();
		child_node->beta = beta;

		nodes[child_name] = child_node;

		//create edge
		CNode* parent = nodes[parent_name];

		parent->add_child(child_node);

		names.push_back(child_name);

		return child_node;

	}


	/**
	 * @param task_lhs task_id on left hand side
	 * @param task_rhs task_id on right hand side
	 * @return similarity between tasks
	 */
	float64_t get_task_similarity(std::string task_lhs, std::string task_rhs)
	{

		CNode* node_lhs = get_node(task_lhs);
		CNode* node_rhs = get_node(task_rhs);

		std::set<CNode*> root_path_lhs = node_lhs->get_path_root();
		std::set<CNode*> root_path_rhs = node_rhs->get_path_root();

		std::set<CNode*> intersection;

		std::set_intersection(root_path_lhs.begin(), root_path_lhs.end(),
							  root_path_rhs.begin(), root_path_rhs.end(),
							  std::inserter(intersection, intersection.end()));

		// sum up weights
		float64_t gamma = 0;
		for (std::set<CNode*>::const_iterator p = intersection.begin(); p != intersection.end(); ++p) {
			gamma += (*p)->beta;
		}

		return gamma;

	}


	int32_t get_num_nodes()
	{
		return (int32_t)names.size();
	}

	float64_t get_node_weight(int32_t idx)
	{
		std::string node_name = names[idx];
		CNode* node = get_node(node_name);
		return node->beta;
	}

	void set_node_weight(int32_t idx, float64_t weight)
	{
		std::string node_name = names[idx];
		CNode* node = get_node(node_name);
		node->beta = weight;
	}

	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "CTaxonomy";
	}


protected:

	CNode* root;
	std::map<std::string, CNode*> nodes;
	std::vector<std::string> names;

};





/** @brief The MultitaskKernel allows Multitask Learning via a modified kernel function.
 *
 * This effectively normalizes the vectors in feature space to norm 1 (see
 * CSqrtDiagKernelNormalizer)
 *
 * \f[
 * k'({\bf x},{\bf x'}) = ...
 * \f]
 */
class CMultitaskKernelTreeNormalizer: public CKernelNormalizer
{

public:

	/** default constructor
	 */
	CMultitaskKernelTreeNormalizer()
	{
	}

	/** default constructor
	 *
	 * @param task_lhs task vector with containing task_id for each example for left hand side
	 * @param task_rhs task vector with containing task_id for each example for right hand side
	 */
	CMultitaskKernelTreeNormalizer(std::vector<std::string> task_lhs,
								   std::vector<std::string> task_rhs,
								   CTaxonomy* tax)
	{

		set_task_vector_lhs(task_lhs);
		set_task_vector_rhs(task_rhs);
		taxonomy = tax;

	}

	/** default destructor */
	virtual ~CMultitaskKernelTreeNormalizer()
	{
	}

	/** initialization of the normalizer
	 * @param k kernel */
	virtual bool init(CKernel* k)
	{
		ASSERT(k);
		int32_t num_lhs = k->get_num_vec_lhs();
		int32_t num_rhs = k->get_num_vec_rhs();
		ASSERT(num_lhs>0);
		ASSERT(num_rhs>0);

		return true;
	}

	/** normalize the kernel value
	 * @param value kernel value
	 * @param idx_lhs index of left hand side vector
	 * @param idx_rhs index of right hand side vector
	 */
	inline virtual float64_t normalize(float64_t value, int32_t idx_lhs,
			int32_t idx_rhs)
	{

		//lookup tasks
		std::string task_idx_lhs = task_vector_lhs[idx_lhs];
		std::string task_idx_rhs = task_vector_rhs[idx_rhs];

		std::cout << task_idx_lhs << ", " << task_idx_rhs << std::endl;

		//lookup similarity
		float64_t task_similarity = get_task_similarity(task_idx_lhs,
				task_idx_rhs);

		//take task similarity into account
		float64_t similarity = value * task_similarity;


		return similarity;

	}

	/** normalize only the left hand side vector
	 * @param value value of a component of the left hand side feature vector
	 * @param idx_lhs index of left hand side vector
	 */
	inline virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs)
	{
		SG_ERROR("normalize_lhs not implemented");
		return 0;
	}

	/** normalize only the right hand side vector
	 * @param value value of a component of the right hand side feature vector
	 * @param idx_rhs index of right hand side vector
	 */
	inline virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs)
	{
		SG_ERROR("normalize_rhs not implemented");
		return 0;
	}

public:

	/** @return vec task vector with containing task_id for each example on left hand side */
	std::vector<std::string> get_task_vector_lhs() const
	{
		return task_vector_lhs;
	}

	/** @param vec task vector with containing task_id for each example */
	void set_task_vector_lhs(std::vector<std::string> vec)
	{
		task_vector_lhs = vec;
	}

	/** @return vec task vector with containing task_id for each example on right hand side */
	std::vector<std::string> get_task_vector_rhs() const
	{
		return task_vector_rhs;
	}

	/** @param vec task vector with containing task_id for each example */
	void set_task_vector_rhs(std::vector<std::string> vec)
	{
		task_vector_rhs = vec;
	}

	/** @param vec task vector with containing task_id for each example */
	void set_task_vector(std::vector<std::string> vec)
	{
		task_vector_lhs = vec;
		task_vector_rhs = vec;
	}

	int32_t get_num_nodes()
	{

		return taxonomy->get_num_nodes();

	}

	float64_t get_node_weight(int32_t idx)
	{

		return taxonomy->get_node_weight(idx);

	}

	void set_node_weight(int32_t idx, float64_t weight)
	{

		taxonomy->set_node_weight(idx, weight);

	}

	/**
	 * @param task_lhs task_id on left hand side
	 * @param task_rhs task_id on right hand side
	 * @return similarity between tasks
	 */
	float64_t get_task_similarity(std::string task_lhs, std::string task_rhs)
	{

		return taxonomy->get_task_similarity(task_lhs, task_rhs);

		/*
		CNode* node_lhs = taxonomy->get_node(task_lhs);
		CNode* node_rhs = taxonomy->get_node(task_rhs);

		std::set<CNode*> root_path_lhs = node_lhs->get_path_root();
		std::set<CNode*> root_path_rhs = node_rhs->get_path_root();

		std::set<CNode*> intersection;

		std::set_intersection(root_path_lhs.begin(), root_path_lhs.end(),
							  root_path_rhs.begin(), root_path_rhs.end(),
							  std::inserter(intersection, intersection.end()));

		// sum up weights
		float64_t gamma = 0;
		for (std::set<CNode*>::const_iterator p = intersection.begin(); p != intersection.end(); ++p) {
			gamma += (*p)->beta;
		}

		return gamma;

		*/
	}

	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "MultitaskKernelTreeNormalizer";
	}

protected:

	/** taxonomy **/
	CTaxonomy* taxonomy;

	/** number of tasks **/
	int32_t num_tasks;

	/** task vector indicating to which task each example on the left hand side belongs **/
	std::vector<std::string> task_vector_lhs;

	/** task vector indicating to which task each example on the right hand side belongs **/
	std::vector<std::string> task_vector_rhs;

};
}
#endif
