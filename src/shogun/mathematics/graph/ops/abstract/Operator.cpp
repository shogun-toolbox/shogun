/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include "Operator.h"

#include <shogun/util/zip_iterator.h>

using namespace shogun;
using namespace shogun::graph::op;

Operator::Operator(const std::shared_ptr<node::Node>& node) : m_node(node)
{
	const auto& shapes = m_node->get_shapes();
	const auto& types = m_node->get_types();

	for (const auto& [shape, type] :
	     zip_iterator(shapes, types))
	{
		m_outputs.push_back(
		    std::make_shared<Storage>(shape, type));
	}
}