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

namespace shogun
{
	class Input;
	// potential entry point to unify backend memory representations?
	// might have to use void*& and store datatype
	class Tensor
	{
	friend class Input;

	protected:
		struct this_is_protected;

	public:
		template <typename T>
		Tensor(const SGVector<T>& vec, this_is_protected) : m_data(vec.vector)
			, m_shape({vec.size()})
			, m_type(get_enum_from_type<T>::type)
		{
		}

		template <typename T>
		Tensor(const SGMatrix<T>& matrix, this_is_protected) : m_data(matrix.matrix)
			, m_shape({matrix.rows(), matrix.cols()})
			, m_type(get_enum_from_type<T>::type)
		{
		}


	public:		
		Tensor(const Shape& shape, element_type type): m_shape(shape), m_type(type)
		{
			m_data = allocator_dispatch(get_size_from_shape(m_shape), m_type);
		}

		const Shape& get_shape() const
		{
			return m_shape;
		}

		size_t size() const
		{
			return get_size_from_shape(m_shape);
		}

		element_type get_type() const
		{
			return m_type;
		}

		void* data()
		{
			return m_data;
		}

		std::string to_string() const;

		friend std::ostream & operator<<(std::ostream&, const std::shared_ptr<Tensor>&);

	protected:

		size_t get_size_from_shape(const Shape& size) const 
		{
			return std::accumulate(
			    size.begin(), size.end(), size_t{1}, std::multiplies{});
		}

		struct this_is_protected {
	       explicit this_is_protected(int) {}
	   	};

	private:
		void* m_data;
		const Shape m_shape;
		const element_type m_type;
		std::vector<std::reference_wrapper<Tensor>> m_inputs;

	};

	std::ostream & operator<<(std::ostream& os, const std::shared_ptr<Tensor>& tensor)
	{
	    return os << tensor->to_string();
	}
} // namespace shogun

#endif