/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNTENSOR_H_
#define SHOGUNTENSOR_H_

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include <shogun/util/enumerate.h>

#include <shogun/mathematics/graph/Allocator.h>
#include <shogun/mathematics/graph/Shape.h>
#include <shogun/mathematics/graph/Types.h>

#include <numeric>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	class Tensor
	{

	friend class Node;

	public:

		template <typename T>
		Tensor(const SGVector<T>& vec) : m_free(false)
		    , m_data(vec.vector)
			, m_shape(Shape{vec.size()})
			, m_type(get_enum_from_type<T>::type)
		{
		}

		template <typename T>
		Tensor(const SGMatrix<T>& matrix) : m_free(false)
		    , m_data(matrix.matrix)
			, m_shape(Shape{matrix.num_rows, matrix.num_cols})
			, m_type(get_enum_from_type<T>::type)
		{
		}

		~Tensor()
		{
			if (m_data != nullptr && m_free)
			{
				switch(m_type)
				{
					case element_type::FLOAT32:
						delete (float32_t*)m_data;
						break;
					case element_type::FLOAT64:
						delete (float32_t*)m_data;
						break;
				}
			}
		}

		Tensor(const Shape& shape, element_type type): m_free(false)
			, m_data(nullptr)
			, m_shape(shape)
			, m_type(type)
		{
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

		void set_shape(const Shape& shape)
		{
			for (auto [idx, original_shape_dim_i, new_shape_dim_i]: enumerate(m_shape, shape))
			{
				if (original_shape_dim_i == Shape::Dynamic)
				{
					m_shape[idx] = new_shape_dim_i;
				}
				else if (original_shape_dim_i != new_shape_dim_i)
				{
					error("Cannot set tensor shape. Shapes {} and {} are incompatible.", m_shape, shape);
				}
			}
		}

		void*& data()
		{
			return m_data;
		}

#ifndef SWIG

		template <typename Container>
		Container as() const
		{
			if constexpr(std::is_same_v<SGVector<typename Container::Scalar>, Container>)
			{
				if (m_shape.size() > 1)
					error("Tried to cast a multidimensional Tensor to a SGVector.");
				if (get_enum_from_type<typename Container::Scalar>::type != m_type)
					error("Type mismatch when casting from Tensor.");
				return SGVector<typename Container::Scalar>((typename Container::Scalar*)m_data, size());
			}
			if constexpr(std::is_same_v<SGMatrix<typename Container::Scalar>, Container>)
			{
				if (get_enum_from_type<typename Container::Scalar>::type != m_type)
					error("Type mismatch when casting from Tensor");
				if (m_shape.size() != 2)
					error("SGMatrix does not support {} dimensions.", m_shape.size());
				return SGMatrix<typename Container::Scalar>((typename Container::Scalar*)m_data, size());
			}
		}

#endif

		[[nodiscard]] size_t get_size_from_shape(const Shape& size) const
		{
			return std::accumulate(
			    size.begin(), size.end(), size_t{1}, std::multiplies{});
		}

	private:
		// whether the memory is owned by the tensor, i.e. should it be deleted by the destructor
		bool m_free;
		void* m_data;
		Shape m_shape;
		const element_type m_type;
	};

} // namespace shogun

#endif