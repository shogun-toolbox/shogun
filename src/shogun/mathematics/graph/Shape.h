/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNSHAPE_H_
#define SHOGUNSHAPE_H_

#include <cstddef>
#include <limits>
#include <vector>
#include <string>
#include <sstream>

namespace shogun
{
	class Shape
	{
	public:
		static constexpr size_t Dynamic = std::numeric_limits<size_t>::max();

		Shape(const std::initializer_list<size_t>& shape) : m_shape(shape)
		{
		}

		Shape(const std::vector<size_t>& shape) : m_shape(shape)
		{
		}

		const std::vector<size_t>& get() const
		{
			return m_shape;
		}

		const size_t size() const
		{
			return m_shape.size();
		}

		auto begin() const
		{
			return m_shape.begin();
		}

		auto end() const
		{
			return m_shape.end();
		}

		std::string to_string() const
		{
			std::stringstream result;
			result << "Shape(";
			auto shape_it = m_shape.begin();
			while(shape_it != m_shape.end())
			{
				if (*shape_it == Dynamic)
					result << "Dynamic";
				else
					result << *shape_it;
				if (std::next(shape_it) != m_shape.end())
					result << ", ";
				shape_it++;
			}
			result << ')';
			return result.str();
		}

		friend std::ostream& operator<<(std::ostream& os, const Shape& shape)
		{
	    	return os << shape.to_string();
		}

	private:
		std::vector<size_t> m_shape;
	};

} // namespace shogun

#endif