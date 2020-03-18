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

#include <shogun/mathematics/graph/Shape.h>
#include <shogun/mathematics/graph/Storage.h>
#include <shogun/mathematics/graph/Types.h>

#include <numeric>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		/* Creates a copy or moves the data to the device
		 */
		inline std::shared_ptr<Storage> device_put(
		    void* ptr, const Shape& shape,
		    const std::shared_ptr<NumberType>& type, const bool copy)
		{
			return std::make_shared<Storage>(
			    ptr, shape, type, copy, Storage::Copy{});
		}

		/* Creates a view to a pointer. The device
		 * memory manager will not call the destructor.
		 */
		inline std::shared_ptr<Storage> device_view(
		    void* ptr, const Shape& shape,
		    const std::shared_ptr<NumberType>& type)
		{
			return Storage::create_view(ptr, shape, type);
		}

		class Tensor
		{
		protected:
			struct Protected
			{
			};

		public:
			friend std::shared_ptr<Tensor>
			from_device(const std::shared_ptr<Storage>& storage);

			/* Creates a copy of a scalar value.
			 * Memory management of the copy is taken over by Tensor.
			 */
			template <
			    typename T, std::enable_if_t<std::is_scalar_v<T>>* = nullptr>
			Tensor(const T& scalar) : m_shape({}), m_type(from<T>())
			{
				m_storage = device_put(new T(scalar), m_shape, m_type, true);
			}

			/* Creates a copy of a SGVector.
			 * Memory management of the copy is taken over by Tensor.
			 */
			template <typename T>
			Tensor(const SGVector<T>& vec)
			    : m_shape(Shape{vec.size()}), m_type(from<T>())
			{
				m_storage = device_put(vec.vector, m_shape, m_type, true);
			}

			/* Creates a copy of a SGMatrix.
			 * Memory management of the copy is taken over by Tensor.
			 */
			template <typename T>
			Tensor(const SGMatrix<T>& matrix)
			    : m_shape(Shape{matrix.num_rows, matrix.num_cols}),
			      m_type(from<T>())
			{
				m_storage = device_put(matrix.matrix, m_shape, m_type, true);
			}

			/* Moves a SGVector to Tensor. The SGVector::vector pointer
			 * is moved to the Tensor. This leaves SGMatrix in a valid,
			 * yet undefined state.
			 * Memory management of the copy is taken over by Tensor.
			 */
			template <typename T>
			Tensor(SGVector<T>&& vec)
			    : m_shape(Shape{vec.size()}), m_type(from<T>())
			{
				m_storage = device_put(
				    std::exchange(vec.vector, nullptr), m_shape, m_type, false);
			}

			/* Moves a SGMatrix to Tensor. The SGMatrix::matrix pointer
			 * is moved to the Tensor. This leaves SGMatrix in a valid,
			 * yet undefined state.
			 * Memory management of the copy is taken over by Tensor.
			 */
			template <typename T>
			Tensor(SGMatrix<T>&& matrix)
			    : m_shape(Shape{matrix.num_rows, matrix.num_cols}),
			      m_type(from<T>())
			{
				m_storage = device_put(
				    std::exchange(matrix.matrix, nullptr), m_shape, m_type,
				    false);
			}

			/* Creates a view to a SGVector. The caller is responsible
			 * with the memory management of the underlying SGVector::vector
			 * pointer. Note that destroying the original SGVector **will**
			 * break the evaluation of the DAG.
			 */
			template <typename T>
			static std::shared_ptr<Tensor> create_view(const SGVector<T>& vec)
			{
				auto result = std::make_shared<Tensor>(
				    Shape{vec.size()}, from<T>(), Protected{});
				result->m_storage = device_view(
				    vec.vector, result->get_shape(), result->get_type());
				return result;
			}

			/* Creates a view to a SGMatrix. The caller is responsible
			 * with the memory management of the underlying SGMatrix::matrix
			 * pointer. Note that destroying the original SGMatrix **will**
			 * break the evaluation of the DAG.
			 */
			template <typename T>
			static std::shared_ptr<Tensor>
			create_view(const SGMatrix<T>& matrix)
			{
				auto result = std::make_shared<Tensor>(
				    Shape{matrix.num_rows, matrix.num_cols}, from<T>(),
				    Protected{});
				result->m_storage = device_view(
				    matrix.matrix, result->get_shape(), result->get_type());
				return result;
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

			[[nodiscard]] const std::shared_ptr<Storage>& storage() const {
				return m_storage;
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
					return *static_cast<Container*>(m_storage);
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
					    (typename Container::Scalar*)m_storage->get_copy(),
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
					    (typename Container::Scalar*)m_storage->get_copy(),
					    m_shape[0], m_shape[1]);
				}
			}
#endif

			Tensor(
			    const Shape& shape, const std::shared_ptr<NumberType>& type,
			    Protected)
			    : m_storage(nullptr), m_shape(shape), m_type(type)
			{
			}

		private:
			[[nodiscard]] size_t get_size_from_shape(const Shape& size) const {
				return std::accumulate(
				    size.begin(), size.end(), size_t{1}, std::multiplies{});
			}

			protected :
			    // the actual data in memory
			    std::shared_ptr<Storage> m_storage;
			// tensor shape
			Shape m_shape;
			// the underlying data type
			std::shared_ptr<NumberType> m_type;
		};

		inline std::shared_ptr<Tensor>
		from_device(const std::shared_ptr<Storage>& storage)
		{
			auto result = std::make_shared<Tensor>(
			    storage->get_shape(), storage->get_type(), Tensor::Protected{});
			result->m_storage = storage;
			return result;
		}
	} // namespace graph
} // namespace shogun

#endif