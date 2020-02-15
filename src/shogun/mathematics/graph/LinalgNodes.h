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

#include <shogun/util/zip_iterator.h>

#define IGNORE_IN_CLASSLIST

namespace shogun {

    // The node classes
    IGNORE_IN_CLASSLIST class Node {
    public:
        Node(const Shape& shape, element_type type): m_output_tensors({std::make_shared<Tensor>(shape, type)}) {}

        Node(const std::vector<Shape>& shapes, const std::vector<element_type>& types)
        {
        	for (const auto& [shape, type]: zip_iterator(shapes, types))
        	{
        		m_output_tensors.push_back(std::make_shared<Tensor>(shape, type));
        	}
        }

        Node(const std::initializer_list<std::shared_ptr<Node>>& nodes,
        	const Shape& shape, element_type type): Node(shape, type) {
        	 m_input_nodes = nodes;
        }

        Node(const std::initializer_list<std::shared_ptr<Node>>& nodes,
        	const std::vector<Shape>& shapes, const std::vector<element_type>& types): Node(shapes, types) {
        	m_input_nodes = nodes;
        }

        virtual ~Node() {
        }

        const std::vector<std::shared_ptr<Node>>& get_input_nodes() const {
            return m_input_nodes;
        }

        const std::vector<std::shared_ptr<Tensor>>& get_tensors() const {
        	return m_output_tensors;
        }

        virtual std::string_view get_operator_name() const = 0;

        virtual std::string to_string() const = 0;

        friend std::ostream& operator<<(std::ostream& os, const Node& node)
		{
	    	return os << node.to_string();
		}

    protected:
        std::vector<std::shared_ptr<Node>> m_input_nodes;
        std::vector<std::shared_ptr<Tensor>> m_output_tensors;
    };
}
#endif