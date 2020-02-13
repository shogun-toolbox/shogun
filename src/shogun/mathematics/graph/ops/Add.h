/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNTENSORADD_H_
#define SHOGUNTENSORADD_H_

#include <shogun/mathematics/graph/LinalgNodes.h>
#include <shogun/mathematics/graph/OperationImplementation.h>

#include <shogun/util/enumerate.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	template <typename T>
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


	template <typename EngineImplementation>
	IGNORE_IN_CLASSLIST class AddImpl: public OperationImpl
	{
	public:
		AddImpl(const std::shared_ptr<Node>& node): OperationImpl(node) {}

		virtual ~AddImpl() {}

		template<typename T>
		void evaluate()
		{
			this->evaluate_implementation();
		}
	};


	IGNORE_IN_CLASSLIST class AddShogun : public AddImpl<AddShogun>
	{
	public:
		AddShogun(const std::shared_ptr<Node>& node) : AddImpl(node)
		{
		}

		void build()
		{
		}

		void evaluate()
		{
			auto add_node = std::static_pointer_cast<Add>(m_abstract_node);

			const auto& node1 = m_abstract_node->get_input_nodes()[0];
			const auto& node2 = m_abstract_node->get_input_nodes()[1];

			add_node->allocate_tensor(
			    rutime_shape_check(node1, node2),
			    node1->get_tensors()[0]->get_type());

			auto input_tensor1 = node1->get_tensors()[0];
			auto input_tensor2 = node2->get_tensors()[0];
			auto output_tensor = add_node->get_tensors()[0];


			// the actual call
			kernel(
			    input_tensor1->data(),
			    input_tensor2->data(),
			    output_tensor->data(), output_tensor->size(),
			    output_tensor->get_type());
		}

	private:

		const Shape& rutime_shape_check(const std::shared_ptr<Node>& node1, 
			const std::shared_ptr<Node>& node2)
		{
			// we don't need to check how many tensors there are
			// the compile time check already did that
			// we also know that the number of dimensions match
			const auto& node1_tensor = node1->get_tensors()[0];
			const auto& node2_tensor = node2->get_tensors()[0];

			for (const auto& [idx, shape1, shape2] : enumerate(
			         node1_tensor->get_shape(),
			         node2_tensor->get_shape()))
			{
				if (shape1 != shape2)
				{
					error("Runtime shape mismatch in dimension {}. Got {} and {}.",
						idx, shape1, shape2);
				}
			}

			return node1_tensor->get_shape();
		}

		void kernel(
		    void* input1, void* input2, void* output, size_t size,
		    element_type type)
		{
			switch (type)
			{
			case element_type::FLOAT32:
				kernel_implementation<
				    get_type_from_enum<element_type::FLOAT32>::type>(
				    input1, input2, output, size);
				break;
			case element_type::FLOAT64:
				kernel_implementation<
				    get_type_from_enum<element_type::FLOAT64>::type>(
				    input1, input2, output, size);
				break;
			}
		}

		template <typename T>
		void kernel_implementation(
		    void* input1, void* input2, void* output, size_t size)
		{
			// if we have SYCL or MSVC we could add parallel execution
			// or just use Eigen here
			std::transform(
			    static_cast<const T*>(input1),
			    static_cast<const T*>(input1) + size,
			    static_cast<const T*>(input2), static_cast<T*>(output),
			    std::plus<T>());
		}
	};

#ifdef USE_NGRAPH
	IGNORE_IN_CLASSLIST class AddNGraph : public AddImpl<AddNGraph>
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