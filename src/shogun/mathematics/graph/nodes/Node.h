/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef LINALGNODES_H_
#define LINALGNODES_H_

#include <shogun/mathematics/graph/Shape.h>
#include <shogun/mathematics/graph/Tensor.h>
#include <shogun/mathematics/graph/Types.h>
#include <shogun/mathematics/graph/nodes/Node.h>

#include <shogun/util/zip_iterator.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		namespace node
		{
			// The node classes
			IGNORE_IN_CLASSLIST class Node
			{
			public:
				using type_info = std::shared_ptr<NumberType>;

				Node(const Shape& shape, element_type type)
				    : m_shapes({shape}), m_types({number_type(type)})
				{
				}

				Node(
				    const std::vector<Shape>& shapes,
				    const std::vector<type_info>& types)
				    : m_shapes(shapes), m_types(types)
				{
				}

				Node(
				    const std::initializer_list<std::shared_ptr<Node>>& nodes,
				    const Shape& shape, element_type type)
				    : Node(shape, type)
				{
					m_input_nodes = nodes;
				}

				Node(
				    const std::initializer_list<std::shared_ptr<Node>>& nodes,
				    const std::vector<Shape>& shapes,
				    const std::vector<type_info>& types)
				    : Node(shapes, types)
				{
					m_input_nodes = nodes;
				}

				virtual ~Node()
				{
				}

				const std::vector<std::shared_ptr<Node>>&
				get_input_nodes() const
				{
					return m_input_nodes;
				}

				// shape of tensor created by this node
				const std::vector<Shape>& get_shapes() const
				{
					return m_shapes;
				}

				// type of tensor created by this node
				const std::vector<type_info>& get_types() const
				{
					return m_types;
				}

				virtual std::string_view get_node_name() const = 0;

				virtual std::string to_string() const = 0;

				friend std::ostream&
				operator<<(std::ostream& os, const Node& node)
				{
					return os << node.to_string();
				}

				virtual bool requires_column_major_conversion() const = 0;

				[[nodiscard]] virtual size_t hash() const
				{
					size_t seed = 0;
					hash_combine(seed, m_types);
					hash_combine(seed, m_shapes);
					hash_combine(seed, m_input_nodes);
					return seed;
				}

			protected:
				std::vector<std::shared_ptr<Node>> m_input_nodes;
				std::vector<Shape> m_shapes;
				std::vector<type_info> m_types;
			};
		} // namespace node
	}     // namespace graph
} // namespace shogun
#endif