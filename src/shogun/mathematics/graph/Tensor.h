/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNTENSOR_H_
#define SHOGUNTENSOR_H_

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include <shogun/mathematics/graph/Allocator.h>
#include <shogun/mathematics/graph/Shape.h>
#include <shogun/mathematics/graph/Types.h>

#include <numeric>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	// potential entry point to unify backend memory representations?
	// might have to use void*& and store datatype
	class Tensor
	{
	friend class Node;

	public:

		template <typename T>
		Tensor(const SGVector<T>& vec) : m_data(vec.vector)
			, m_shape(Shape{vec.size()})
			, m_type(get_enum_from_type<T>::type)
		{
		}

		template <typename T>
		Tensor(const SGMatrix<T>& matrix) : m_data(matrix.matrix)
			, m_shape(Shape{matrix.num_rows, matrix.num_cols})
			, m_type(get_enum_from_type<T>::type)
		{
		}

		Tensor(const Shape& shape, element_type type): m_shape(shape), m_type(type)
		{
			m_data = allocator_dispatch(get_size_from_shape(m_shape), m_type);
		}

		[[nodiscard]] const Shape& get_shape() const
		{
			return m_shape;
		}

		[[nodiscard]] size_t size() const
		{
			return get_size_from_shape(m_shape);
		}

		[[nodiscard]] element_type get_type() const
		{
			return m_type;
		}

		[[nodiscard]] std::string to_string() const;

		friend std::ostream & operator<<(std::ostream& os, const std::shared_ptr<Tensor>& tensor)
		{
	    	return os << tensor->to_string();
		}

		void*& data()
		{
			return m_data;
		}

		[[nodiscard]] size_t get_size_from_shape(const Shape& size) const
		{
			return std::accumulate(
			    size.begin(), size.end(), size_t{1}, std::multiplies{});
		}

	private:
		// m_data is temporary, only the execution engine can assume
		// that it is valid. A user should only have access to data
		// through the Result class, obtained from Graph::evaluate(...);
		void* m_data;
		const Shape m_shape;
		const element_type m_type;
	};

} // namespace shogun

#endif