/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include "utils.h"
#include <shogun/util/enumerate.h>

using namespace shogun;
using namespace shogun::graph;


void shogun::graph::op::assert_dynamic_shape(const Shape& input_shape1, const Shape& input_shape2)
{
	for (auto [idx, shape1, shape2] : enumerate(
	         input_shape1, input_shape2))
	{
		if (shape1 != shape2)
		{
			error(
			    "Runtime shape mismatch in dimension {}. "
			    "Got "
			    "{} and {}.",
			    idx, shape1, shape2);
		}
		if (shape1 == Shape::Dynamic)
		{
			error("Could not infer runtime shape.");
		}
	}
}