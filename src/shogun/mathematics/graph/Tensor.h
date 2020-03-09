/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNTENSOR_H_
#define SHOGUNTENSOR_H_

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/memory.h>

#include <shogun/util/enumerate.h>

#include <shogun/mathematics/graph/Shape.h>
#include <shogun/mathematics/graph/Types.h>
#include <shogun/mathematics/graph/ops/abstract/ShogunStorage.h>

#include <numeric>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		inline std::shared_ptr<ShogunStorage> device_put(
		    void* ptr, const Shape& shape,
		    const std::shared_ptr<NumberType>& type)
		{
			return std::shared_ptr<ShogunStorage>(
			    new ShogunStorage(ptr, shape, type));
		}

		class Tensor
		{
		public:
			friend std::shared_ptr<Tensor>
			from_device(const std::shared_ptr<ShogunStorage>& storage);

			template <typename T>
			Tensor(const T& scalar) : m_shape({}), m_type(from<T>())
			{
				m_data = device_put(new T(scalar), m_shape, m_type);
			}

			template <typename T>
			Tensor(const SGVector<T>& vec)
			    : m_shape(Shape{vec.size()}), m_type(from<T>())
			{
				m_data = device_put(vec.vector, m_shape, m_type);
			}

			template <typename T>
			Tensor(const SGMatrix<T>& matrix)
			    : m_shape(Shape{matrix.num_rows, matrix.num_cols}),
			      m_type(from<T>())
			{
				m_data = device_put(matrix.matrix, m_shape, m_type);
			}

			[[nodiscard]] const Shape& get_shape() const { return m_shape; }

			    [[nodiscard]] size_t size_in_bytes() const
			{
				return size() * m_type->size();
			}

			[[nodiscard]] size_t size() const {
				return get_size_from_shape(m_shape);
			}

			    [[nodiscard]] std::shared_ptr<NumberType> get_type() const
			{
				return m_type;
			}

			[[nodiscard]] std::string to_string() const;

			friend std::ostream&
			operator<<(std::ostream& os, const std::shared_ptr<Tensor>& tensor)
			{
				return os << tensor->to_string();
			}

			[[nodiscard]] const std::shared_ptr<ShogunStorage>& data() const {
				return m_data;
			}

#ifndef SWIG

			template <typename Container>
			Container as() const
			{
				if constexpr (std::is_arithmetic_v<Container>)
				{
					if (size() > 1)
						error("Cannot cast a non scalar representation to a "
						      "scalar type.");
					return *static_cast<Container*>(m_data);
				}

				else if constexpr (std::is_same_v<
				                       SGVector<typename Container::Scalar>,
				                       Container>)
				{
					if (m_shape.size() > 1)
						error("Tried to cast a multidimensional Tensor to a "
						      "SGVector.");
					if (from<typename Container::Scalar>() != m_type)
						error("Type mismatch when casting from Tensor.");
					return Container(
					    (typename Container::Scalar*)m_data->get_copy(),
					    size());
				}
				else if constexpr (std::is_same_v<
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
					    (typename Container::Scalar*)m_data->get_copy(),
					    m_shape[0], m_shape[1]);
				}
			}
#endif
		protected:
			Tensor() = default;

		private:
			[[nodiscard]] size_t get_size_from_shape(const Shape& size) const {
				return std::accumulate(
				    size.begin(), size.end(), size_t{1}, std::multiplies{});
			}

			protected :
			    // the actual data in memory
			    std::shared_ptr<ShogunStorage> m_data;
			// tensor shape
			Shape m_shape;
			// the underlying data type
			std::shared_ptr<NumberType> m_type;
		};

		inline std::shared_ptr<Tensor>
		from_device(const std::shared_ptr<ShogunStorage>& storage)
		{
			auto result = std::shared_ptr<Tensor>(new Tensor());
			result->m_data = storage;
			result->m_shape = storage->m_shape;
			result->m_type = storage->m_type;
			return result;
		}
	} // namespace graph
} // namespace shogun

#endif