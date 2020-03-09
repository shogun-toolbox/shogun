/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNSHAPE_H_
#define SHOGUNSHAPE_H_

#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <shogun/io/SGIO.h>
#include <shogun/util/zip_iterator.h>

namespace shogun
{
	namespace graph
	{
		class Tensor;

		class Shape
		{
		public:
			using shape_type = int64_t;

			static constexpr shape_type Dynamic = -1;

			Shape() = default;

			Shape(const std::initializer_list<shape_type>& shape)
			    : m_shape(shape)
			{
			}

			Shape(const std::vector<shape_type>& shape) : m_shape(shape)
			{
			}

			Shape(std::vector<shape_type>&& shape)
			    : m_shape(std::move(shape)){}

			          [[nodiscard]] shape_type size() const
			{
				return m_shape.size();
			}

			const shape_type& operator[](size_t idx) const
			{
				return m_shape[idx];
			}

			shape_type& operator[](size_t idx)
			{
				return m_shape[idx];
			}

			bool operator==(const Shape& other)
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

			bool partial_compare(size_t idx, shape_type other) const
			{
				if (is_scalar())
					error("Cannot do Shape::partial_compare with scalar shape "
					      "representations");
				if (m_shape[idx] == Shape::Dynamic || other == Shape::Dynamic)
					return true;
				return m_shape[idx] == other;
			}

			bool is_static() const
			{
				for (const auto& dim : m_shape)
				{
					if (dim == Shape::Dynamic)
						return false;
				}
				return true;
			}

			bool is_scalar() const
			{
				return m_shape.size() == 0;
			}

			Shape switch_major() const
			{
				if (m_shape.size() < 2)
					return *this;
				else
				{
					auto result = m_shape;
					std::swap(result[0], result[1]);
					return Shape{result};
				}
			}

			[[nodiscard]] std::vector<Shape::shape_type>::const_iterator
			begin() const { return m_shape.begin(); }

			    [[nodiscard]] std::vector<Shape::shape_type>::const_iterator
			    end() const
			{
				return m_shape.end();
			}

			[[nodiscard]] std::string to_string() const {
				std::stringstream result;
				result << "Shape(";
				auto shape_it = m_shape.begin();
				while (shape_it != m_shape.end())
				{
					if (*shape_it == Dynamic)
						result << "?";
					else
						result << *shape_it;
					if (std::next(shape_it) != m_shape.end())
						result << ", ";
					shape_it++;
				}
				result << ')';
				return result.str();
			}

			friend std::ostream&
			operator<<(std::ostream& os, const Shape& shape)
			{
				return os << shape.to_string();
			}

		private:
			std::vector<shape_type> m_shape;
		};
	} // namespace graph
} // namespace shogun

#endif