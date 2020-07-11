/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Viktor Gal, Soeren Sonnenburg, Heiko Strathmann,
 *          Yuyu Zhang, Bjoern Esser, Sanuj Sharma
 */

#ifndef _MULTITASKKERNELTREENORMALIZER_H___
#define _MULTITASKKERNELTREENORMALIZER_H___

#include <shogun/lib/config.h>

#include <shogun/transfer/multitask/MultitaskKernelMklNormalizer.h>
#include <shogun/kernel/Kernel.h>
#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <deque>
#include <vector>

namespace shogun
{

/** @brief A Node is an element of a Taxonomy, which is used to describe hierarchical
 *	structure between tasks.
 *
 */
class Node: public SGObject
{
public:
	using NodeSet = std::set<std::shared_ptr<Node>>;
	/** default constructor
	 */
    Node()
    {
        parent = NULL;
        beta = 1.0;
        node_id = 0;
    }

    ~Node() override
    {
    }

    /** get a list of all ancestors of this node
     *  @return set of Nodes
	 */
    Node::NodeSet get_path_root()
    {
        Node::NodeSet nodes_on_path;
        std::shared_ptr<Node> node = shared_from_this()->as<Node>();
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
        std::deque<std::shared_ptr<Node>> grey_nodes;
        grey_nodes.push_back(shared_from_this()->as<Node>());

        while(grey_nodes.size() > 0)
        {

            auto current_node = grey_nodes.front();
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
    void add_child(std::shared_ptr<Node >node)
    {
        node->parent = shared_from_this()->as<Node>();
        this->children.push_back(node);
    }

    /** @return object name */
    const char *get_name() const override
    {
        return "Node";
    }

    /** @return boolean indicating, whether this node is a leaf */
    bool is_leaf() const noexcept
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
	std::shared_ptr<Node> parent;

	/** list of child nodes **/
	std::vector<std::shared_ptr<Node>> children;

	/** identifier of node **/
	int32_t node_id;

};


/** @brief Taxonomy is used to describe hierarchical
 *	structure between tasks.
 *
 */
class Taxonomy
{

public:

	/** default constructor
	 */
	Taxonomy()
	{
		root = std::make_shared<Node>();
		nodes.push_back(root);

		name2id = std::map<std::string, int32_t>();
		name2id["root"] = 0;
	}

	virtual ~Taxonomy()
	{
	}

	/**
	 *  @param task_id task identifier
	 *  @return node with id task_id
	 */
	std::shared_ptr<Node> get_node(int32_t task_id) const 
	{
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
	std::shared_ptr<Node> add_node(const std::string& parent_name, 
		const std::string& child_name, float64_t beta)
	{
		if (child_name=="")	error("child_name empty");
		if (parent_name=="") error("parent_name empty");


		auto child_node = std::make_shared<Node>();

		child_node->beta = beta;

		nodes.push_back(child_node);
		int32_t id = nodes.size()-1;

		name2id[child_name] = id;

		child_node->setNode_id(id);


		//create edge
		auto parent = nodes[name2id[parent_name]];

		parent->add_child(child_node);

		return child_node;
	}

	/** translates name to id
	 *  @param name name of task
	 *  @return id
	 */
	int32_t get_id(const std::string& name) 
	{
		return name2id[name];
	}

	/** given two nodes, compute the intersection of their ancestors
	 *  @param node_lhs node of left hand side
	 *  @param node_rhs node of right hand side
	 *  @return intersection of the two sets of ancestors
	 */
	Node::NodeSet intersect_root_path(const std::shared_ptr<Node>& node_lhs, 
		const std::shared_ptr<Node>& node_rhs) const
	{
		const auto& root_path_lhs = node_lhs->get_path_root();
		const auto& root_path_rhs = node_rhs->get_path_root();

		Node::NodeSet intersection;

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
	float64_t compute_node_similarity(int32_t task_lhs, int32_t task_rhs) const
	{
		auto node_lhs = get_node(task_lhs);
		auto node_rhs = get_node(task_rhs);

		// compute intersection of paths to root
		const auto& intersection = intersect_root_path(node_lhs, node_rhs);

		// sum up weights
		float64_t gamma = 0;
		for (Node::NodeSet::const_iterator p = intersection.begin(); p != intersection.end(); ++p)
		{
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
			task_histogram[it->first] = task_histogram[it->first] / static_cast<float64_t>(task_vector_lhs.size());
		}
	}

	/** @return number of nodes */
	int32_t get_num_nodes() const noexcept
	{
		return nodes.size();
	}

	/** @return number of leaves */
	int32_t get_num_leaves() const
	{
		int32_t num_leaves = 0;

		for (int32_t i=0; i<get_num_nodes(); i++)
		{
			if (get_node(i)->is_leaf()==true)
			{
				num_leaves++;
			}
		}

		return num_leaves;
	}

	/** @return weight of node with identifier idx */
	float64_t get_node_weight(int32_t idx) const
	{
		auto node = get_node(idx);
		return node->beta;
	}

	/**
	 *  @param idx node id
	 *  @param weight weight to set
	 */
	void set_node_weight(int32_t idx, float64_t weight)
	{
		auto node = get_node(idx);
		node->beta = weight;
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "Taxonomy";
	}

	/** @return mapping from name to id */
	std::map<std::string, int32_t> get_name2id() const noexcept 
	{
		return name2id;
	}

	/**
	 *  translate name to id
	 *  @param name node name
	 *  @return id
	 */
	int32_t get_id_by_name(const std::string& name) const
	{
		return name2id.at(name);
	}

protected:
	/** root */
	std::shared_ptr<Node> root;
	/** name 2 id */
	std::map<std::string, int32_t> name2id;
	/** nodes */
	std::vector<std::shared_ptr<Node>> nodes;
	/** task histogram */
	std::map<int32_t, float64_t> task_histogram;
};

/** @brief The MultitaskKernel allows Multitask Learning via a modified kernel function based on taxonomy.
 *
 */
class MultitaskKernelTreeNormalizer: public MultitaskKernelMklNormalizer
{
public:

	/** default constructor
	 */
	MultitaskKernelTreeNormalizer() : MultitaskKernelMklNormalizer()
	{
	}

	/** default constructor
	 *
	 * @param task_lhs task vector with containing task_id for each example for left hand side
	 * @param task_rhs task vector with containing task_id for each example for right hand side
	 * @param tax taxonomy
	 */
	MultitaskKernelTreeNormalizer(const std::vector<std::string>& task_lhs,
								  const std::vector<std::string>& task_rhs,
								  Taxonomy tax): MultitaskKernelTreeNormalizer()
	{
		taxonomy = tax;
		set_task_vector_lhs(task_lhs);
		set_task_vector_rhs(task_rhs);

		num_nodes = taxonomy.get_num_nodes();

		dependency_matrix = SGMatrix<float64_t>(num_nodes, num_nodes);

		update_cache();
	}

	/** default destructor */
	~MultitaskKernelTreeNormalizer() override
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
	float64_t normalize(float64_t value, int32_t idx_lhs, int32_t idx_rhs) const override
	{
		//lookup tasks
		const auto& task_idx_lhs = task_vector_lhs[idx_lhs];
		const auto& task_idx_rhs = task_vector_rhs[idx_rhs];

		//lookup similarity
		float64_t task_similarity = get_node_similarity(task_idx_lhs, task_idx_rhs);
		//float64_t task_similarity = taxonomy.compute_node_similarity(task_idx_lhs, task_idx_rhs);

		//take task similarity into account
		float64_t similarity = (value / scale) * task_similarity;

		return similarity;
	}

	/** normalize only the left hand side vector
	 * @param value value of a component of the left hand side feature vector
	 * @param idx_lhs index of left hand side vector
	 */
	float64_t normalize_lhs(float64_t value, int32_t idx_lhs) const override
	{
		error("normalize_lhs not implemented");
		return 0;
	}

	/** normalize only the right hand side vector
	 * @param value value of a component of the right hand side feature vector
	 * @param idx_rhs index of right hand side vector
	 */
	float64_t normalize_rhs(float64_t value, int32_t idx_rhs) const override
	{
		error("normalize_rhs not implemented");
		return 0;
	}


	/** @param vec task vector with containing task_id for each example */
	void set_task_vector_lhs(const std::vector<std::string>& vec)
	{
		task_vector_lhs.clear();

		for (int32_t i = 0; i < vec.size(); ++i)
		{
			task_vector_lhs.push_back(taxonomy.get_id(vec[i]));
		}

		//update task histogram
		taxonomy.update_task_histogram(task_vector_lhs);

	}

	/** @param vec task vector with containing task_id for each example */
	void set_task_vector_rhs(const std::vector<std::string>& vec)
	{
		task_vector_rhs.clear();

		for (int32_t i = 0; i < vec.size(); ++i)
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
	int32_t get_num_betas() const noexcept override
	{
		return taxonomy.get_num_nodes();
	}

	/**
	 * @param idx id of weight
	 * @return weight of node with given id */
	float64_t get_beta(int32_t idx) const override
	{
		return taxonomy.get_node_weight(idx);
	}

	/**
	 * @param idx id of weight
	 * @param weight weight of node with given id */
	void set_beta(int32_t idx, float64_t weight) override
	{
		taxonomy.set_node_weight(idx, weight);

		update_cache();
	}

	/**
	 * @param node_lhs node_id on left hand side
	 * @param node_rhs node_id on right hand side
	 * @return similarity between nodes
	 */
	float64_t get_node_similarity(int32_t node_lhs, int32_t node_rhs) const
	{
		ASSERT(node_lhs < num_nodes && node_lhs >= 0)
		ASSERT(node_rhs < num_nodes && node_rhs >= 0)

		return dependency_matrix(node_lhs, node_rhs);
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

		dependency_matrix(node_lhs, node_rhs) = similarity;
	}

	/** @return object name */
	const char* get_name() const override
	{
		return "MultitaskKernelTreeNormalizer";
	}

	/** casts kernel normalizer to multitask kernel tree normalizer
	 * @param n kernel normalizer to cast
	 */
	std::shared_ptr<MultitaskKernelTreeNormalizer> KernelNormalizerToMultitaskKernelTreeNormalizer(std::shared_ptr<KernelNormalizer> n)
	{
		return std::dynamic_pointer_cast<MultitaskKernelTreeNormalizer>(n);
	}

protected:
	/** taxonomy **/
	Taxonomy taxonomy;

	/** number of tasks **/
	int32_t num_nodes;

	/** task vector indicating to which task each example on the left hand side belongs **/
	std::vector<int32_t> task_vector_lhs;

	/** task vector indicating to which task each example on the right hand side belongs **/
	std::vector<int32_t> task_vector_rhs;

	/** MxM matrix encoding similarity between tasks **/
	SGMatrix<float64_t> dependency_matrix;
};
}
#endif
