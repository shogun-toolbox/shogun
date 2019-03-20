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
	/*
	 * A type agnostic representation of parameter trees that uses a CSGObject
	 * instance to generate parameter combinations.
	 *
	 */
	class ParameterNode: public CSGObject
	{

	public:
		/**
		 * Parameter node constructor with model
		 * @param model
		 * @param param_properties
		 */
		ParameterNode(CSGObject& model);

		ParameterNode(CSGObject& model, ParameterProperties param_propeties);

#ifndef SWIG
		/**
		 * The default constructor
		 */
		ParameterNode() = default;

		/**
		 * Creates a node that mimics the CSGObject structure
		 * and places it in the tree using the given name.
		 * @param name
		 * @param obj
		 */
		virtual void create_node(const std::string& name, CSGObject* obj);
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

		virtual const char* get_name() const
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

		virtual std::string to_string() const;

	protected:
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
		std::map<std::string, std::shared_ptr<ParameterNode>> m_nodes;
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

	class GridParameters : public ParameterNode
	{

	public:
		GridParameters(CSGObject& model);

		virtual void create_node(const std::string& name, CSGObject* obj);

		const char* get_name() const final
		{
			return "GridParameters";
		}

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

		ParameterNode* get_next() final;
		ParameterNode* get_current() final;

		bool set_param_helper(const std::string& param, const Any& value) final;

		bool check_child_node_done()
		{
			return m_node_complete;
		}

		bool is_complete();

		void set_node_complete(bool flag)
		{
			m_node_complete = flag;
		}

	private:
		bool first;

	private:
		/** tracks current node being held fixed */
		std::map<std::string, std::shared_ptr<ParameterNode>>::iterator
		    m_current_node;
		/** tracks current parameter being iterated over*/
		std::map<std::string, Any>::iterator m_current_param;
		/** current iterator of a given parameter */
		std::unordered_map<std::string, Any> m_param_iter;
		/** iterator begin of a given parameter */
		std::unordered_map<std::string, Any> m_param_begin;
		/** iterator end of a given parameter	 */
		std::unordered_map<std::string, Any> m_param_end;

		bool m_node_complete;
	};

	class GridSearch : public GridParameters
	{
		GridSearch() = default;

		GridSearch(CSGObject& model) : GridParameters(model)
		{
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
