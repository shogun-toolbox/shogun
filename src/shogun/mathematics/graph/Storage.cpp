/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/mathematics/graph/Storage.h>
#include <shogun/util/zip_iterator.h>

using namespace shogun;
using namespace shogun::graph;


void Storage::set_static_shape(const Shape& shape)
{
	if (m_shape.size() != shape.size())
	{
		error(
		    "Mismatch in the number of dimensions, expected "
		    "{}, "
		    "but got {}",
		    m_shape.size(), shape.size());
	}

	for (const auto& [original_shape_dim_i, new_shape_dim_i] :
	     zip_iterator(m_shape, shape))
	{
		if (original_shape_dim_i != new_shape_dim_i)
		{
			error(
			    "Cannot set tensor shape. Shapes {} and {} are "
			    "incompatible.",
			    m_shape, shape);
		}
	}
	m_shape = shape;
}

void Storage::set_shape(const Shape& shape)
{
	if (!shape.is_static())
		error("Cannot set dynamic shape in storage.");
	if (m_shape.size() != shape.size())
	{
		error(
		    "Mismatch in the number of dimensions, expected "
		    "{}, "
		    "but got {}",
		    m_shape.size(), shape.size());
	}

	for (const auto& [original_shape_dim_i, new_shape_dim_i] :
	     zip_iterator(m_shape, shape))
	{
		if (original_shape_dim_i == Shape::Dynamic)
			continue;
		if (original_shape_dim_i != new_shape_dim_i)
		{
			error(
			    "Cannot set tensor shape. Shapes {} and {} are "
			    "incompatible.",
			    m_shape, shape);
		}
	}
	m_shape = shape;
}