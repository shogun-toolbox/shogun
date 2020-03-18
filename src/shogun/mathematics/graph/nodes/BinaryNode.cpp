/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include "BinaryNode.h"
#include <shogun/util/enumerate.h>

using namespace shogun;
using namespace shogun::graph;
using namespace shogun::graph::node;


Shape BaseBinaryNode::same_shape_binary_op(
    const Shape& node1_shape, const Shape& node2_shape)
{
	std::vector<Shape::shape_type> output_shape_vector;

	for (const auto& [idx, shape1, shape2] :
	     enumerate(node1_shape, node2_shape))
	{
		if (shape1 == shape2)
		{
			output_shape_vector.push_back(shape1);
		}
		else if (
		    shape1 == Shape::Dynamic &&
		    shape2 == Shape::Dynamic)
		{
			output_shape_vector.push_back(Shape::Dynamic);
		}
		else if (
		    shape1 != Shape::Dynamic &&
		    shape2 != Shape::Dynamic && shape1 != shape2)
		{
			// this is a mismatch, it can't possible go well at
			// runtime
			error(
			    "Shape mismatch in dimension {} when comparing "
			    "{} and {}",
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
			error("BinaryNode: Unexpected path, contact a dev "
			      "or raise an "
			      "issue!");
		}
	}

	return Shape{output_shape_vector};
}