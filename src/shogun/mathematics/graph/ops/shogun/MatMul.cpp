/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include "MatMul.h"

#include <shogun/util/enumerate.h>

using namespace shogun;
using namespace shogun::graph;
using namespace shogun::graph::op;


void MatMulShogun::runtime_checks_and_allocation(
    const std::shared_ptr<Storage>& input1,
    const std::shared_ptr<Storage>& input2,
    const bool transpose_a, const bool transpose_b)
{
	const auto& shape_a = input1->get_shape();
	const auto& shape_b = input2->get_shape();

	auto reduction_axis_a =
	    std::static_pointer_cast<node::MatMul>(m_node)
	        ->get_reduction_axis_a();
	auto reduction_axis_b =
	    std::static_pointer_cast<node::MatMul>(m_node)
	        ->get_reduction_axis_b();

	if (transpose_a)
		reduction_axis_a = std::abs(
		    static_cast<int64_t>(reduction_axis_a) - 1);

	if (transpose_b)
		reduction_axis_b = std::abs(
		    static_cast<int64_t>(reduction_axis_b) - 1);

	if (shape_a[reduction_axis_a] != shape_b[reduction_axis_b])
	{
		error(
		    "Runtime MatMul shape mismatch. "
		    "shapes {} and {} not aligned: {} (dim {}) != "
		    "{} (dim {})",
		    shape_a.to_string(), shape_b.to_string(),
		    shape_a[reduction_axis_a], reduction_axis_a,
		    shape_b[reduction_axis_b], reduction_axis_b);
	}

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

	m_outputs[0]->allocate_storage(Shape{output_shape_vector});
}