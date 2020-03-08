/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNTENSOR_H_
#define SHOGUNTENSOR_H_

#include <shogun/lib/memory.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include <shogun/util/enumerate.h>

#include <shogun/mathematics/graph/Shape.h>
#include <shogun/mathematics/graph/Types.h>

#include <numeric>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		class Tensor
		{
		public:
			template <typename T>
			Tensor(const T& scalar)
			    : m_free(true), m_data(new T(scalar)), m_shape({}),
			      m_type(from<T>())
			{
			}

			template <typename T>
			Tensor(const SGVector<T>& vec)
			    : m_free(false), m_data(vec.vector), m_shape(Shape{vec.size()}),
			      m_type(from<T>())
			{
			}

			template <typename T>
			Tensor(const SGMatrix<T>& matrix)
			    : m_free(false), m_data(matrix.matrix),
			      m_shape(Shape{matrix.num_rows, matrix.num_cols}),
			      m_type(from<T>())
			{
			}

			~Tensor()
			{
				if (m_data != nullptr && m_free)
				{
					SG_ALIGNED_FREE(m_data);
					m_data = nullptr;
				}
			}

			Tensor(const Shape& shape, const std::shared_ptr<NumberType>& type)
			    : m_free(false), m_data(nullptr), m_shape(shape), m_type(type)
			{
			}

			void allocate_tensor(const Shape& shape)
			{
				if (m_data != nullptr)
					error("Tensor already owns data!");
				set_shape(shape);
				m_data = sg_aligned_malloc(size_in_bytes(), alignment::container_alignment);
				m_free = true;
			}

			[[nodiscard]] const Shape& get_shape() const { return m_shape; }

			[[nodiscard]] Shape& get_shape()
			{
				return m_shape;
			}

			[[nodiscard]] size_t size_in_bytes() const {
				return size() * m_type->size();
			}

			[[nodiscard]] size_t size() const
			{
				return get_size_from_shape(m_shape);
			}

			[[nodiscard]] std::shared_ptr<NumberType> get_type() const { return m_type; }

			[[nodiscard]] std::string to_string() const;

			friend std::ostream&
			operator<<(std::ostream& os, const std::shared_ptr<Tensor>& tensor)
			{
				return os << tensor->to_string();
			}

			void set_shape(const Shape& shape)
			{
				if (m_shape.size() != shape.size())
				{
					error(
					    "Mismatch in the number of dimensions, expected {}, "
					    "but got {}",
					    m_shape.size(), shape.size());
				}

				for (auto [idx, original_shape_dim_i, new_shape_dim_i] :
				     enumerate(m_shape, shape))
				{
					if (original_shape_dim_i == Shape::Dynamic)
					{
						m_shape[idx] = new_shape_dim_i;
					}
					else if (original_shape_dim_i != new_shape_dim_i)
					{
						error(
						    "Cannot set tensor shape. Shapes {} and {} are "
						    "incompatible.",
						    m_shape, shape);
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
				if constexpr (std::is_same_v<
				                  SGVector<typename Container::Scalar>,
				                  Container>)
				{
					if (m_shape.size() > 1)
						error("Tried to cast a multidimensional Tensor to a "
						      "SGVector.");
					if (from<typename Container::Scalar>() != m_type)
						error("Type mismatch when casting from Tensor.");
					return Container(
					    (typename Container::Scalar*)m_data, size(), false);
				}
				if constexpr (std::is_same_v<
				                  SGMatrix<typename Container::Scalar>,
				                  Container>)
				{
					if (from<typename Container::Scalar>() != m_type)
						error("Type mismatch when casting from Tensor");
					if (m_shape.size() != 2)
						error(
						    "SGMatrix does not support {} dimensions.",
						    m_shape.size());
					return Container(
					    (typename Container::Scalar*)m_data, m_shape[0],
					    m_shape[1], false);
				}
			}

#endif

			[[nodiscard]] size_t get_size_from_shape(const Shape& size) const {
				return std::accumulate(
				    size.begin(), size.end(), size_t{1}, std::multiplies{});
			}

			private :
			    // whether the memory is owned by the tensor, i.e. should it be
			    // deleted by the destructor
			    bool m_free;
			// the actual data in memory
			void* m_data;
			// tensor shape
			Shape m_shape;
			std::shared_ptr<NumberType> m_type;
		};
	} // namespace graph
} // namespace shogun

#endif