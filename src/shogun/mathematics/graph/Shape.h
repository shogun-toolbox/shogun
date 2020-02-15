/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNSHAPE_H_
#define SHOGUNSHAPE_H_

#include <cstddef>
#include <limits>
#include <utility>
#include <vector>
#include <string>
#include <sstream>

namespace shogun
{
	class Tensor;

	class Shape
	{
	public:

		friend class Tensor;

		using shape_type = int64_t;

		static constexpr shape_type Dynamic = -1;

		Shape(std::initializer_list<shape_type> shape) : m_shape(shape)
		{
		}

		Shape(std::vector<shape_type> shape) : m_shape(std::move(shape))
		{
		}

		[[nodiscard]] shape_type size() const
		{
			return m_shape.size();
		}

		[[nodiscard]] shape_type operator[](size_t idx) const
		{
			return m_shape[idx];
		}

		[[nodiscard]] auto begin() const
		{
			return m_shape.begin();
		}

		[[nodiscard]] auto end() const
		{
			return m_shape.end();
		}

		[[nodiscard]] std::string to_string() const
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

	protected:

		void set_dimension(shape_type new_value, shape_type dim)
		{
			m_shape[dim] = new_value;
		}

	private:
		std::vector<shape_type> m_shape;
	};

} // namespace shogun

#endif