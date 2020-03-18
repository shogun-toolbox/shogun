/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include "Dot.h"
#include <shogun/util/enumerate.h>

using namespace shogun;
using namespace shogun::graph;
using namespace shogun::graph::node;

element_type Dot::check_type_compatible(
    const std::shared_ptr<Node>& A,
    const std::shared_ptr<Node>& B)
{
	const auto& node1_types = A->get_types();
	const auto& node2_types = B->get_types();

	if (node1_types.size() > 1)
		error(
		    "Expected first node to have only one output "
		    "tensor, but got {}",
		    node1_types.size());

	if (node2_types.size() > 1)
		error(
		    "Expected second node to have only one output "
		    "tensor, but got {}",
		    node2_types.size());

	if (node1_types[0] != node2_types[0])
		error("Expected types to be the same");

	return node1_types[0]->type();
}


std::tuple<Shape, size_t, size_t> Dot::check_shape_compatible(
    const Shape& shape_a, const Shape& shape_b)
{
	size_t reduction_axis_a =
	    shape_a.size() < 1 ? 0 : shape_a.size() - 1;
	size_t reduction_axis_b =
	    shape_b.size() <= 1 ? 0 : shape_b.size() - 2;

	// one of the values is a scalar
	if (shape_a.is_scalar())
	{
		return std::make_tuple(
		    shape_b, reduction_axis_a, reduction_axis_b);
	}
	else if (shape_b.is_scalar())
	{
		return std::make_tuple(
		    shape_a, reduction_axis_a, reduction_axis_b);
	}

	if (!shape_a.partial_compare(
	        reduction_axis_a, shape_b[reduction_axis_b]))
		error(
		    "shapes {} and {} not aligned: {} (dim {}) != {} "
		    "(dim {})",
		    shape_a.to_string(), shape_b.to_string(),
		    shape_a[reduction_axis_a], reduction_axis_a,
		    shape_b[reduction_axis_b], reduction_axis_b);

	std::vector<Shape::shape_type> output_shape_vector;

	for (const auto& [idx, el] : enumerate(shape_a))
	{
		if (idx != reduction_axis_a)
			output_shape_vector.push_back(el);
	}

	for (const auto& [idx, el] : enumerate(shape_b))
	{
		if (idx != reduction_axis_b)
			output_shape_vector.push_back(el);
	}

	return std::make_tuple(
	    Shape{output_shape_vector}, reduction_axis_a,
	    reduction_axis_b);
}