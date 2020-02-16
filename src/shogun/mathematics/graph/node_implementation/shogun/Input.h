/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_DETAIL_INPUTNGRAPH_H_
#define SHOGUN_DETAIL_INPUTSHOGUN_H_

#include <shogun/mathematics/graph/node_implementation/NodeImplementation.h>
#include <shogun/mathematics/graph/nodes/Input.h>
#include <shogun/mathematics/graph/ops/shogun/Input.h>


namespace shogun
{
	namespace graph
	{
		namespace detail
		{
			namespace shogun
			{
				IGNORE_IN_CLASSLIST class InputShogun
				    : public RuntimeNodeTemplate<node::Input, OutputNode>
				{
				public:
					InputShogun() : RuntimeNodeTemplate()
					{
					}

					std::string_view get_runtime_node_name() const final
					{
						return "Input";
					}

					[[nodiscard]] std::shared_ptr<OutputNode> build_input(const std::shared_ptr<node::Node>& node) const
					{
						return std::make_shared<OutputNode>(std::make_shared<op::InputShogun>(node));
					}

					[[nodiscard]] std::shared_ptr<OutputNode> build_implementation(const std::shared_ptr<node::Node>& node) const final
					{
						error("Input nodes use Input::build_input(node) instead.");
						return nullptr;
					}
				};
			} // namespace shogun
		}     // namespace detail
	}         // namespace graph
} // namespace shogun

#endif
