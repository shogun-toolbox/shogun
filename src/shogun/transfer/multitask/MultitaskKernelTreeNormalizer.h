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

#include <shogun/transfer/multitask/MultitaskKernelMklNormalizer.h>
#include <shogun/kernel/Kernel.h>
#include <algorithm>
#include <map>
#include <set>
#include <deque>
#include <vector>

namespace shogun
{

/** @brief A CNode is an element of a CTaxonomy, which is used to describe hierarchical
 *	structure between tasks.
 *
 */
class CNode: public CSGObject
{
public:
	/** default constructor
	 */
    CNode()
    {
        parent = NULL;
        beta = 1.0;
        node_id = 0;
    }

    virtual ~CNode()
    {
	for (size_t i = 0; i < children.size(); i++)
		delete children[i];
    }

    /** get a list of all ancestors of this node
     *  @return set of CNodes
	 */
    std::set<CNode*> get_path_root()
    {
        std::set<CNode*> nodes_on_path = std::set<CNode*>();
        CNode *node = this;
        while (node != NULL) {
            nodes_on_path.insert(node);
            node = node->parent;
        }
        return nodes_on_path;
    }

    /** get a list of task ids at the leaves below the current node
	 *  @return list of task ids
	 */
    std::vector<int32_t> get_task_ids_below()
    {

        std::vector<int32_t> task_ids;
        std::deque<CNode*> grey_nodes;
        grey_nodes.push_back(this);

        while(grey_nodes.size() > 0)
        {

            CNode *current_node = grey_nodes.front();
            grey_nodes.pop_front();

            for(int32_t i = 0; i!=int32_t(current_node->children.size()); i++){
                grey_nodes.push_back(current_node->children[i]);
            }

            if(current_node->is_leaf()){
	task_ids.push_back(current_node->getNode_id());
            }
        }

        return task_ids;
    }

    /** add child to current node
	 *  @param node child node
	 */
    void add_child(CNode *node)
    {
        node->parent = this;
        this->children.push_back(node);
    }

    /** @return object name */
    virtual const char *get_name() const
    {
        return "Node";
    }

    /** @return boolean indicating, whether this node is a leaf */
    bool is_leaf()
	{
		return children.empty();

	}

    /** @return node id of current node */
    int32_t getNode_id() const
    {
        return node_id;
    }

    /** @param node_idx node id for current node */
    void setNode_id(int32_t node_idx)
    {
        this->node_id = node_idx;
    }

    /** parameter of node **/
	float64_t beta;

protected:

	/** parent node **/
	CNode* parent;

	/** list of child nodes **/
	std::vector<CNode*> children;

	/** identifier of node **/
	int32_t node_id;

};


/** @brief CTaxonomy is used to describe hierarchical
 *	structure between tasks.
 *
 */
class CTaxonomy : public CSGObject
{

public:

	/** default constructor
	 */
	CTaxonomy() : CSGObject()
	{
		root = new CNode();
		nodes.push_back(root);

		name2id = std::map<std::string, int32_t>();
		name2id["root"] = 0;
	}

	virtual ~CTaxonomy()
	{
		for (size_t i = 0; i < nodes.size(); i++)
			delete nodes[i];
		nodes.clear();
		name2id.clear();
		task_histogram.clear();
	}

	/**
	 *  @param task_id task identifier
	 *  @return node with id task_id
	 */
	CNode* get_node(int32_t task_id) {
		return nodes[task_id];
	}

	/** set root weight
	 *  @param beta weight
	 */
	void set_root_beta(float64_t beta)
	{
		nodes[0]->beta = beta;
	}

	/** inserts additional node into taxonomy
	 *  @param parent_name name of parent
	 *  @param child_name name of child
	 *  @param beta weight of child
	 */
	CNode* add_node(std::string parent_name, std::string child_name, float64_t beta)
	{
		if (child_name=="")	SG_ERROR("child_name empty")
		if (parent_name=="") SG_ERROR("parent_name empty")


		CNode* child_node = new CNode();

		child_node->beta = beta;

		nodes.push_back(child_node);
		int32_t id = nodes.size()-1;

		name2id[child_name] = id;

		child_node->setNode_id(id);


		//create edge
		CNode* parent = nodes[name2id[parent_name]];

		parent->add_child(child_node);

		return child_node;
	}

	/** translates name to id
	 *  @param name name of task
	 *  @return id
	 */
	int32_t get_id(std::string name) {
		return name2id[name];
	}

	/** given two nodes, compute the intersection of their ancestors
	 *  @param node_lhs node of left hand side
	 *  @param node_rhs node of right hand side
	 *  @return intersection of the two sets of ancestors
	 */
	std::set<CNode*> intersect_root_path(CNode* node_lhs, CNode* node_rhs)
	{

		std::set<CNode*> root_path_lhs = node_lhs->get_path_root();
		std::set<CNode*> root_path_rhs = node_rhs->get_path_root();

		std::set<CNode*> intersection;

		std::set_intersection(root_path_lhs.begin(), root_path_lhs.end(),
							  root_path_rhs.begin(), root_path_rhs.end(),
							  std::inserter(intersection, intersection.end()));

		return intersection;

	}

	/**
	 * @param task_lhs task_id on left hand side
	 * @param task_rhs task_id on right hand side
	 * @return similarity between tasks
	 */
	float64_t compute_node_similarity(int32_t task_lhs, int32_t task_rhs)
	{

		CNode* node_lhs = get_node(task_lhs);
		CNode* node_rhs = get_node(task_rhs);

		// compute intersection of paths to root
		std::set<CNode*> intersection = intersect_root_path(node_lhs, node_rhs);

		// sum up weights
		float64_t gamma = 0;
		for (std::set<CNode*>::const_iterator p = intersection.begin(); p != intersection.end(); ++p) {

			gamma += (*p)->beta;
		}

		return gamma;

	}

	/** keep track of how many elements each task has
	 * @param task_vector_lhs vector of task ids for examples
	 */
	void update_task_histogram(std::vector<int32_t> task_vector_lhs) {

		//empty map
		task_histogram.clear();


		//fill map with zeros
		for (std::vector<int32_t>::const_iterator it=task_vector_lhs.begin(); it!=task_vector_lhs.end(); it++)
		{
			task_histogram[*it] = 0.0;
		}

		//fill map
		for (std::vector<int32_t>::const_iterator it=task_vector_lhs.begin(); it!=task_vector_lhs.end(); it++)
		{
			task_histogram[*it] += 1.0;
		}

		//compute fractions
		for (std::map<int32_t, float64_t>::const_iterator it=task_histogram.begin(); it!=task_histogram.end(); it++)
		{
			task_histogram[it->first] = task_histogram[it->first] / float64_t(task_vector_lhs.size());
		}

	}

	/** @return number of nodes */
	int32_t get_num_nodes()
	{
		return (int32_t)(nodes.size());
	}

	/** @return number of leaves */
	int32_t get_num_leaves()
	{
		int32_t num_leaves = 0;

		for (int32_t i=0; i!=get_num_nodes(); i++)
		{
			if (get_node(i)->is_leaf()==true)
			{
				num_leaves++;
			}
		}

		return num_leaves;
	}

	/** @return weight of node with identifier idx */
	float64_t get_node_weight(int32_t idx)
	{
		CNode* node = get_node(idx);
		return node->beta;
	}

	/**
	 *  @param idx node id
	 *  @param weight weight to set
	 */
	void set_node_weight(int32_t idx, float64_t weight)
	{
		CNode* node = get_node(idx);
		node->beta = weight;
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "Taxonomy";
	}

	/** @return mapping from name to id */
	std::map<std::string, int32_t> get_name2id() {
		return name2id;
	}

	/**
	 *  translate name to id
	 *  @param name node name
	 *  @return id
	 */
	int32_t get_id_by_name(std::string name)
	{
		return name2id[name];
	}


protected:

	/** root */
	CNode* root;
	/** name 2 id */
	std::map<std::string, int32_t> name2id;
	/** nodes */
	std::vector<CNode*> nodes;
	/** task histogram */
	std::map<int32_t, float64_t> task_histogram;

};

/** @brief The MultitaskKernel allows Multitask Learning via a modified kernel function based on taxonomy.
 *
 */
class CMultitaskKernelTreeNormalizer: public CMultitaskKernelMklNormalizer
{

public:

	/** default constructor
	 */
	CMultitaskKernelTreeNormalizer() : CMultitaskKernelMklNormalizer()
	{
	}

	/** default constructor
	 *
	 * @param task_lhs task vector with containing task_id for each example for left hand side
	 * @param task_rhs task vector with containing task_id for each example for right hand side
	 * @param tax taxonomy
	 */
	CMultitaskKernelTreeNormalizer(std::vector<std::string> task_lhs,
								   std::vector<std::string> task_rhs,
								   CTaxonomy tax) : CMultitaskKernelMklNormalizer()
	{

		taxonomy = tax;
		set_task_vector_lhs(task_lhs);
		set_task_vector_rhs(task_rhs);

		num_nodes = taxonomy.get_num_nodes();

		dependency_matrix = std::vector<float64_t>(num_nodes * num_nodes);

		update_cache();
	}


	/** default destructor */
	virtual ~CMultitaskKernelTreeNormalizer()
	{
	}


	/** update cache */
	void update_cache()
	{


		for (int32_t i=0; i!=num_nodes; i++)
		{
			for (int32_t j=0; j!=num_nodes; j++)
			{

				float64_t similarity = taxonomy.compute_node_similarity(i, j);
				set_node_similarity(i,j,similarity);

			}

		}
	}



	/** normalize the kernel value
	 * @param value kernel value
	 * @param idx_lhs index of left hand side vector
	 * @param idx_rhs index of right hand side vector
	 */
	virtual float64_t normalize(float64_t value, int32_t idx_lhs, int32_t idx_rhs)
	{
		//lookup tasks
		int32_t task_idx_lhs = task_vector_lhs[idx_lhs];
		int32_t task_idx_rhs = task_vector_rhs[idx_rhs];

		//lookup similarity
		float64_t task_similarity = get_node_similarity(task_idx_lhs, task_idx_rhs);
		//float64_t task_similarity = taxonomy.compute_node_similarity(task_idx_lhs, task_idx_rhs);

		//take task similarity into account
		float64_t similarity = (value/scale) * task_similarity;


		return similarity;
	}

	/** normalize only the left hand side vector
	 * @param value value of a component of the left hand side feature vector
	 * @param idx_lhs index of left hand side vector
	 */
	virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs)
	{
		SG_ERROR("normalize_lhs not implemented")
		return 0;
	}

	/** normalize only the right hand side vector
	 * @param value value of a component of the right hand side feature vector
	 * @param idx_rhs index of right hand side vector
	 */
	virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs)
	{
		SG_ERROR("normalize_rhs not implemented")
		return 0;
	}


	/** @param vec task vector with containing task_id for each example */
	void set_task_vector_lhs(std::vector<std::string> vec)
	{

		task_vector_lhs.clear();

		for (int32_t i = 0; i != (int32_t)(vec.size()); ++i)
		{
			task_vector_lhs.push_back(taxonomy.get_id(vec[i]));
		}

		//update task histogram
		taxonomy.update_task_histogram(task_vector_lhs);

	}

	/** @param vec task vector with containing task_id for each example */
	void set_task_vector_rhs(std::vector<std::string> vec)
	{

		task_vector_rhs.clear();

		for (int32_t i = 0; i != (int32_t)(vec.size()); ++i)
		{
			task_vector_rhs.push_back(taxonomy.get_id(vec[i]));
		}

	}

	/** @param vec task vector with containing task_id for each example */
	void set_task_vector(std::vector<std::string> vec)
	{
		set_task_vector_lhs(vec);
		set_task_vector_rhs(vec);
	}

	/** @return number of parameters/weights */
	int32_t get_num_betas()
	{

		return taxonomy.get_num_nodes();

	}

	/**
	 * @param idx id of weight
	 * @return weight of node with given id */
	float64_t get_beta(int32_t idx)
	{

		return taxonomy.get_node_weight(idx);

	}

	/**
	 * @param idx id of weight
	 * @param weight weight of node with given id */
	void set_beta(int32_t idx, float64_t weight)
	{

		taxonomy.set_node_weight(idx, weight);

		update_cache();

	}


	/**
	 * @param node_lhs node_id on left hand side
	 * @param node_rhs node_id on right hand side
	 * @return similarity between nodes
	 */
	float64_t get_node_similarity(int32_t node_lhs, int32_t node_rhs)
	{

		ASSERT(node_lhs < num_nodes && node_lhs >= 0)
		ASSERT(node_rhs < num_nodes && node_rhs >= 0)

		return dependency_matrix[node_lhs * num_nodes + node_rhs];

	}

	/**
	 * @param node_lhs node_id on left hand side
	 * @param node_rhs node_id on right hand side
	 * @param similarity similarity between nodes
	 */
	void set_node_similarity(int32_t node_lhs, int32_t node_rhs,
			float64_t similarity)
	{

		ASSERT(node_lhs < num_nodes && node_lhs >= 0)
		ASSERT(node_rhs < num_nodes && node_rhs >= 0)

		dependency_matrix[node_lhs * num_nodes + node_rhs] = similarity;

	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "MultitaskKernelTreeNormalizer";
	}

	/** casts kernel normalizer to multitask kernel tree normalizer
	 * @param n kernel normalizer to cast
	 */
	CMultitaskKernelTreeNormalizer* KernelNormalizerToMultitaskKernelTreeNormalizer(CKernelNormalizer* n)
	{
		return dynamic_cast<CMultitaskKernelTreeNormalizer*>(n);
	}

protected:
	/** taxonomy **/
	CTaxonomy taxonomy;

	/** number of tasks **/
	int32_t num_nodes;

	/** task vector indicating to which task each example on the left hand side belongs **/
	std::vector<int32_t> task_vector_lhs;

	/** task vector indicating to which task each example on the right hand side belongs **/
	std::vector<int32_t> task_vector_rhs;

	/** MxM matrix encoding similarity between tasks **/
	std::vector<float64_t> dependency_matrix;
};
}
#endif
