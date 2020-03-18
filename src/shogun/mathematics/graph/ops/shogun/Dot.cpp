/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include "Dot.h"

#include <shogun/util/enumerate.h>

using namespace shogun;
using namespace shogun::graph;
using namespace shogun::graph::op;

void DotShogun::runtime_checks_and_allocation(
    const std::shared_ptr<Storage>& input1,
    const std::shared_ptr<Storage>& input2)
{
	const auto& shape_a = input1->get_shape();
	const auto& shape_b = input2->get_shape();

	const auto& reduction_axis_a =
	    std::static_pointer_cast<node::Dot>(m_node)
	        ->get_reduction_axis_a();
	const auto& reduction_axis_b =
	    std::static_pointer_cast<node::Dot>(m_node)
	        ->get_reduction_axis_b();

	if (shape_a.is_scalar())
	{
		m_outputs[0]->allocate_storage(shape_b);
	}
	else if (shape_b.is_scalar())
	{
		m_outputs[0]->allocate_storage(shape_a);
	}
	else
	{
		if (shape_a[reduction_axis_a] !=
		    shape_b[reduction_axis_b])
		{
			error(
			    "Runtime Dot shape mismatch. "
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

		m_outputs[0]->allocate_storage(
		    Shape{output_shape_vector});
	}
}