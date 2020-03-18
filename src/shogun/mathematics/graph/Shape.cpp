/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/mathematics/graph/Shape.h>
#include <shogun/util/zip_iterator.h>

using namespace shogun;
using namespace shogun::graph;


bool Shape::operator==(const Shape& other) const
{
	for (const auto& [el1, el2] : zip_iterator(*this, other))
	{
		if (el1 == Shape::Dynamic || el2 == Shape::Dynamic)
			continue;
		if (el1 != el2)
			return false;
	}
	return true;
}