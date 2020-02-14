/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNTENSORADD_H_
#define SHOGUNTENSORADD_H_

#include <shogun/mathematics/graph/LinalgNodes.h>
#include <shogun/util/enumerate.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	template <typename T1, typename T2>
	class AddImpl;
	class AddShogun;

	IGNORE_IN_CLASSLIST class Add : public Node
	{
		friend class AddShogun;

	public:
		Add(const std::shared_ptr<Node>& node1,
		    const std::shared_ptr<Node>& node2): Node({node1, node2},
		    check_shape_compatible(node1, node2),
		    check_type_compatible(node1, node2)
		    )
		{
		}

		std::string to_string() const override
		{
			return fmt::format("Add(shape={}, type={})", get_tensors()[0]->get_shape(), get_tensors()[0]->get_type());
		}

		std::string_view get_operator_name() const override
		{
			return "Add";
		}

	private:
		element_type check_type_compatible(
		    const std::shared_ptr<Node>& node1,
		    const std::shared_ptr<Node>& node2)
		{
			auto node1_tensors = node1->get_tensors();
			auto node2_tensors = node2->get_tensors();

			if (node1_tensors.size() > 1)
				error(
				    "Expected first node to have only one output tensor, but "
				    "got {}",
				    node1_tensors.size());

			if (node2_tensors.size() > 1)
				error(
				    "Expected second node to have only one output tensor, but "
				    "got {}",
				    node2_tensors.size());

			if (node1_tensors[0]->get_type() != node2_tensors[0]->get_type())
				error("Expected types to be the same");

			return node1_tensors[0]->get_type();
		}

		Shape check_shape_compatible(
		    const std::shared_ptr<Node>& node1,
		    const std::shared_ptr<Node>& node2)
		{
			auto node1_tensors = node1->get_tensors();
			auto node2_tensors = node2->get_tensors();

			if (node1_tensors.size() > 1)
				error(
				    "Expected first node to have only one output tensor, but "
				    "got {}",
				    node1_tensors.size());

			if (node2_tensors.size() > 1)
				error(
				    "Expected second node to have only one output tensor, but "
				    "got {}",
				    node2_tensors.size());

			if (node1_tensors[0]->get_shape().size() !=
			    node2_tensors[0]->get_shape().size())
			{
				error(
				    "Number of dimension mismatch between {} and {}.", node1,
				    node2);
			}

			std::vector<size_t> output_shape_vector;

			for (const auto& [idx, shape1, shape2] : enumerate(
			         node1_tensors[0]->get_shape(),
			         node2_tensors[0]->get_shape()))
			{
				if (shape1 == shape2)
				{
					output_shape_vector.push_back(shape1);
				}
				else if (shape1 == Shape::Dynamic && shape2 == Shape::Dynamic)
				{
					output_shape_vector.push_back(Shape::Dynamic);
				}
				else if (
				    shape1 != Shape::Dynamic && shape2 != Shape::Dynamic &&
				    shape1 != shape2)
				{
					// this is a mismatch, it can't possible go well at runtime
					error(
					    "Shape mismatch in dimension {} when comparing {} and "
					    "{}",
					    idx, shape1, shape2);
				}
				else if (shape1 == Shape::Dynamic)
				{
					// shape2 is more restrictive so pick that one
					output_shape_vector.push_back(shape2);
				}
				else if (shape2 == Shape::Dynamic)
				{
					// shape1 is more restrictive so pick that one
					output_shape_vector.push_back(shape1);
				}
				else
				{
					error("Unexpected path: contact a dev or raise an issue!");
				}
			}

			return Shape{output_shape_vector};
		}

		void allocate_tensor(const Shape& shape, element_type type)
		{
			m_output_tensors[0]->data() = allocator_dispatch(
			    m_output_tensors[0]->get_size_from_shape(shape), type);
		}
	};

#ifdef USE_NGRAPH
	IGNORE_IN_CLASSLIST class AddNGraph : public AddImpl<AddNGraph, OperatorNGraphBackend>
	{
	public:
		AddNGraph(const std::shared_ptr<Node>& node) : AddImpl(node)
		{
		}

		void build()
		{
			m_ngraph_node = std::make_shared<ngraph::op::Add>();
		}

		void evaluate()
		{
		}
	};
#endif
} // namespace shogun

#endif