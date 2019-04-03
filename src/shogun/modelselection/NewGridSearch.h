/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_PARAMETER_TREE_H
#define SHOGUN_PARAMETER_TREE_H

#include <shogun/base/AnyParameter.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/type_case.h>
#include <shogun/machine/Machine.h>
#include <unordered_set>

namespace shogun
{
	/**
	 * A type agnostic representation of parameter trees that uses a CSGObject
	 * instance to generate parameter combinations.
	 *
	 */
	class ParameterNode
	{

	public:
		/**
		 * Parameter node constructor with model
		 * @param model
		 * @param param_properties
		 */
		explicit ParameterNode(CSGObject& model);

		explicit ParameterNode(const Some<CSGObject>& model)
		    : ParameterNode(*model.get())
		{
		}

		ParameterNode(CSGObject& model, ParameterProperties param_propeties);

#ifndef SWIG
		/**
		 * The default constructor
		 */
		ParameterNode() = default;
#endif

		/**
		 * Attach a new node at a given location in the tree
		 * defined by the parameter name
		 * @param name
		 * @param node
		 */
		ParameterNode* attach(
		    const std::string& param,
		    const std::shared_ptr<ParameterNode>& node);

		/**
		 * Attach a new node at a given location in the tree
		 * defined by the parameter name
		 * @param name
		 * @param node
		 */
		ParameterNode* attach(const std::string& param, ParameterNode* node);

		/**
		 * Attach a new value at a given location in the tree
		 * defined by the parameter name
		 * @param name
		 * @param node
		 */
		template <typename T>
		ParameterNode* attach(const std::string& param, T value)
		{
			if (!set_param_helper(param, make_any(value)))
				SG_SERROR("Could not attach %s", param.c_str());
			return this;
		}

		/**
		 * Returns the next parameter combination.
		 *
		 * @return
		 */
		ParameterNode* next_combination()
		{
			return get_next();
		}

		virtual const char* get_name() const noexcept
		{
			return "ParameterNode";
		}

		/**
		 * Returns the name of the node which is
		 * infered by the CSGObject
		 * @return
		 */
		const char* get_parent_name() const
		{
			return m_parent->get_name();
		}

		std::string to_string() const noexcept;

		/**
		 * Replaces a node in a specific position of the tree node.
		 * A tree node can contain several nodes which represent
		 * different objects that can be used by the parent object.
		 *
		 * @param node_name
		 * @param index
		 * @param node
		 */
		void replace_node(
		    const std::string& node_name, size_t index,
		    const std::shared_ptr<ParameterNode>& node);

		size_t n_nodes()
		{
			return m_nodes.size();
		}

	protected:
		/**
		 * Creates a node that mimics the CSGObject structure
		 * and places it in the tree using the given name.
		 * This is meant to be only used inside constructors,
		 * as it doesn't check if the node should exist.
		 * @param name
		 * @param obj
		 */
		virtual void create_node(const std::string& name, CSGObject* obj);
		/**
		 * Internal method to set the value (with Any type) of a node
		 *
		 * @param param
		 * @param value
		 * @return
		 */
		virtual bool
		set_param_helper(const std::string& param, const Any& value);
		/**
		 * Internal method to set the node within a node
		 *
		 * @param param
		 * @param value
		 * @return
		 */
		bool set_param_helper(
		    const std::string& param,
		    const std::shared_ptr<ParameterNode>& value);
		/**
		 * Internal method to get the next combination. In the base class
		 * this is the current set, as it is the only parameter combination
		 * available.
		 *
		 * @return
		 */
		virtual ParameterNode* get_next()
		{
			return get_current();
		}
		/**
		 * Internal method to get the current parameter combination
		 * @return
		 */
		virtual ParameterNode* get_current();

		/** vector of child nodes */
		std::map<std::string, std::vector<std::shared_ptr<ParameterNode>>>
		    m_nodes;
		/** pointer to object that is being mimicked */
		std::shared_ptr<CSGObject> m_parent;
		/** internal mapping of params */
		std::map<std::string, Any> m_param_mapping;

		/**
		 * Definition of the delimiter used to refer to nested parameters.
		 * For example, when delimiter is '::' obj1::param1 refers to param1 in
		 * obj1
		 */
		static const std::string delimiter;

		/**
		 * Internal method to get the string representation of each value in the
		 * tree
		 * @param ss
		 * @param visitor
		 */
		void to_string_helper(
		    std::stringstream& ss, std::unique_ptr<AnyVisitor>& visitor) const
		    noexcept;

		/**
		 * Function that returns the CSGObject with the parameters
		 * given by the tree.
		 *
		 * @param tree
		 * @return
		 */
		static CSGObject* to_object(const std::shared_ptr<ParameterNode>& tree);
	};

	/**
	 * GridParameters represents each node in a ParameterNode
	 * as a vector, and calculates the cartesian product of all
	 * nodes in the tree.
	 * Each combination is generated based on the previously generated tree
	 * using recursive calls to nodes and iterating over the parameter
	 * maps. The state of each node and parameter map is stored in the form
	 * of an iterator.
	 */
	class GridParameters : public ParameterNode
	{

	public:
		GridParameters(CSGObject& model);

		const char* get_name() const noexcept override
		{
			return "GridParameters";
		}

		/**
		 * Attach a new node at a given location in the tree
		 * defined by the parameter name
		 * @param name
		 * @param node
		 */
		GridParameters* attach(const std::string& param, GridParameters* node);

		/**
		 * Attach a new SGVector at a given location in the tree
		 * defined by the parameter name
		 * @param name
		 * @param node
		 */
		template <typename T>
		GridParameters* attach(const std::string& param, SGVector<T> values)
		{
			if (!set_param_helper(param, make_any(values)))
				SG_SERROR("Could not attach %s", param.c_str());
			return this;
		}

	protected:
		/**
		 * Resets the internal state of the node.
		 */
		void reset();

		/**
		 * Creates a new node based on a CSGObject
		 * @param name
		 * @param obj
		 */
		void create_node(const std::string& name, CSGObject* obj) override;

		/**
		 * Gets the next ParameterNode representation
		 * of this instance
		 * @return
		 */
		ParameterNode* get_next() final;

		/**
		 * Gets the current ParameterNode representation
		 * of this instance
		 * @return
		 */
		ParameterNode* get_current() final;

		bool set_param_helper(const std::string& param, const Any& value) final;

		/**
		 * Checks if all nodes below this one have been completed.
		 *
		 * @return
		 */
		bool is_complete();

		/**
		 * Checks if all nested nodes in m_nodes have been completed.
		 * @return
		 */
		bool is_inner_complete(const std::string&);

		/**
		 * Set node complete status. This is needed if a parent
		 * needs to reset the complete status of a child node.
		 *
		 * @param flag
		 */
		void set_node_complete(bool flag)
		{
			m_node_complete = flag;
		}

		/**
		 * Child node complete status getter.
		 *
		 * @return
		 */
		bool check_child_node_done()
		{
			return m_node_complete;
		}

	private:
		/** if it is the first iteration of the node */
		bool m_first;
		/** tracks current node being iterated over. A node can have several
		 * internal nodes */
		std::map<std::string, std::vector<std::shared_ptr<ParameterNode>>>::
		    iterator m_current_node;
		/** tracks current internal node being iterated over */
		std::vector<std::shared_ptr<ParameterNode>>::iterator
		    m_current_internal_node;
		/** tracks current parameter being iterated over*/
		std::map<std::string, Any>::iterator m_current_param;
		/** current iterator of a given parameter */
		std::unordered_map<std::string, Any> m_param_iter;
		/** iterator begin of a given parameter */
		std::unordered_map<std::string, Any> m_param_begin;
		/** iterator end of a given parameter */
		std::unordered_map<std::string, Any> m_param_end;
		/** whether the node is complete, i.e. all parameter combinations have
		 * been generated */
		bool m_node_complete;
	};

	class GridSearch : public GridParameters
	{

		GridSearch(CSGObject& model) : GridParameters(model)
		{
		}

		const char* get_name() const noexcept override
		{
			return "GridSearch";
		}

		void train()
		{
			while (!is_complete())
			{
				auto next = std::shared_ptr<ParameterNode>(get_next());
				auto* machine = ParameterNode::to_object(next)->as<CMachine>();
				machine->train();
				SG_UNREF(machine);
			}
		}

	private:
		SGVector<float64_t> m_scores;
	};
} // namespace shogun

#endif // SHOGUN_PARAMETER_TREE_H
