/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef NODE_RUNTIME_IMPLEMENTATION_H_
#define NODE_RUNTIME_IMPLEMENTATION_H_

#include <shogun/io/SGIO.h>
#include <shogun/mathematics/graph/nodes/Node.h>

#include <memory>
#include <string_view>
#include <vector>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace detail
		{

			IGNORE_IN_CLASSLIST class RuntimeNode
			{
			public:
				RuntimeNode() = default;
				virtual ~RuntimeNode(){};

				virtual std::string_view get_runtime_node_name() const = 0;
			};

			template <typename AbstractNodeType, typename SpecialisedNode>
			IGNORE_IN_CLASSLIST class RuntimeNodeTemplate : public RuntimeNode
			{

			public:
				using abstract_node_type = AbstractNodeType;

				RuntimeNodeTemplate() = default;

				virtual ~RuntimeNodeTemplate(){}

				    [[nodiscard]] std::shared_ptr<SpecialisedNode> build(
				        const std::vector<std::shared_ptr<SpecialisedNode>>&
				            input_nodes,
				        const std::shared_ptr<node::Node>& node)
				{
					m_input_nodes = input_nodes;
					return build_implementation(node);
				}

				// virtual void evaluate() = 0;
				[[nodiscard]] virtual std::shared_ptr<SpecialisedNode>
				build_implementation(
				    const std::shared_ptr<node::Node>&) const = 0;

			protected:
				std::vector<std::shared_ptr<SpecialisedNode>> m_input_nodes;
			};

#define REGISTER_OP_FACTORY(opr, NODE)                                         \
	REGISTER_OP_UNIQ_HELPER(__COUNTER__, opr, NODE)
#define REGISTER_OP_UNIQ_HELPER(ctr, opr, NODE) REGISTER_OP_UNIQ(ctr, opr, NODE)
#define REGISTER_OP_UNIQ(ctr, opr, NODE)                                       \
	static auto register_opf##ctr SG_ATTRIBUTE_UNUSED =                        \
	    opr.emplace(std::type_index(typeid(NODE::abstract_node_type)), []() {  \
		    return std::make_shared<NODE>();                                   \
	    })
#define REGISTER_OP(NODE) REGISTER_OP_FACTORY(OperatorRegistry(), NODE)
		} // namespace detail
	}     // namespace graph
} // namespace shogun

#endif