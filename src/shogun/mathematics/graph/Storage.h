/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_OPERATOR_STORAGE_H_
#define SHOGUN_OPERATOR_STORAGE_H_

#include <shogun/lib/memory.h>
#include <shogun/mathematics/graph/Shape.h>
#include <shogun/mathematics/graph/Types.h>
#include <shogun/util/zip_iterator.h>

#include <numeric>
#include <utility>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			class ReshapeShogun;
		}
		/* A storage abstraction
		 * The user should never see this. This is purely for
		 * the backend to allocate storage. The user should use
		 * the tensor class exclusively, in order to not mess
		 * up shapes and so on.
		 *
		 */
		class Storage
		{
			/* Managed storage. Its only job is to keep track of memory
			 * and how to delete and allocate it.
			 * It's unaware of what it actually owns. This is pretty much
			 * a wrapper around std::shared_ptr for aligned memory.
			 * Should this be thread safe?
			 */
			class MemoryManager
			{
				struct Move
				{
				};
				struct NonOwning
				{
				};

				using deleted_unique_ptr =
				    std::unique_ptr<void, std::function<void(void*)>>;

			public:
				MemoryManager(
				    size_t size, const std::shared_ptr<NumberType>& type)
				{
					void* data = sg_aligned_malloc(
					    size_in_bytes(size, type),
					    alignment::container_alignment);
					m_internal_data = deleted_unique_ptr(data, [](void* ptr) {
						if (ptr)
							SG_ALIGNED_FREE(ptr);
					});
				}

				MemoryManager(void* data)
				    : m_internal_data(data, [](void* ptr) {
					      if (ptr)
						      SG_ALIGNED_FREE(ptr);
				      })
				{
				}

				/* Copies a pointer to MemoryManager
				 */
				static std::unique_ptr<MemoryManager> copy_from(
				    void* ptr, size_t size,
				    const std::shared_ptr<NumberType>& type)
				{
					auto result = std::make_unique<MemoryManager>(size, type);
					sg_memcpy(
					    result->m_internal_data.get(), ptr,
					    size_in_bytes(size, type));
					return result;
				}

				/* Moves ownership of the pointer to MemoryManager
				 */
				static std::unique_ptr<MemoryManager> move_from(
				    void* ptr, size_t size,
				    const std::shared_ptr<NumberType>& type)
				{
					return std::make_unique<MemoryManager>(
					    ptr, size, type, Move{});
				}

				/* Generates a non owning view of the pointer
				 */
				static std::unique_ptr<MemoryManager> view(
				    void* ptr, size_t size,
				    const std::shared_ptr<NumberType>& type)
				{
					return std::make_unique<MemoryManager>(
					    ptr, size, type, NonOwning{});
				}

				void
				realloc(size_t old_size, size_t size, const std::shared_ptr<NumberType>& type)
				{
					void* new_mem = sg_aligned_realloc(
						m_internal_data.get(), size_in_bytes(old_size, type),
						size_in_bytes(size, type), alignment::container_alignment);
					// sg_aligned_realloc throws ShogunException if it failed
					m_internal_data.reset(new_mem);
				}

				void get_copy(
				    void*& dst, size_t size,
				    const std::shared_ptr<NumberType>& type) const
				{
					sg_memcpy(
					    dst, m_internal_data.get(), size_in_bytes(size, type));
				}

				[[nodiscard]] void* get_copy(
				    size_t size,
				    const std::shared_ptr<NumberType>& type) const {
					void* dst = sg_aligned_malloc(
					    size_in_bytes(size, type),
					    alignment::container_alignment);
					sg_memcpy(
					    dst, m_internal_data.get(), size_in_bytes(size, type));
					return dst;
				}

				    [[nodiscard]] static size_t size_in_bytes(
				        size_t size, const std::shared_ptr<NumberType>& type)
				{
					return size * type->size();
				}

				deleted_unique_ptr m_internal_data;

				/* Constructor called by move_from.
				 * The memory manager acquires data.
				 */
				MemoryManager(
				    void* data, size_t size,
				    const std::shared_ptr<NumberType>& type, Move)
				{
					m_internal_data = deleted_unique_ptr(data, [](void* ptr) {
						if (ptr)
							SG_ALIGNED_FREE(ptr);
					});
				}

				/* Constructor called by view.
				 * The memory manager does not own the data,
				 * only has a view into it.
				 */
				MemoryManager(
				    void* data, size_t size,
				    const std::shared_ptr<NumberType>& type, NonOwning)
				{
					m_internal_data =
					    deleted_unique_ptr(data, [](void* ptr) { /*noop*/ });
				}
			};

		protected:
			// tags that give access to public constructors to shared_ptr
			// but callee needs to be a friend
			struct View
			{
			};
			struct Copy
			{
			};

		public:
			friend class op::ReshapeShogun;
			friend std::shared_ptr<Storage> device_put(
			    void*, const Shape&, const std::shared_ptr<NumberType>&, bool);
			friend std::shared_ptr<Tensor>
			from_device(const std::shared_ptr<Storage>& storage);
			friend class NGraph;

			Storage(const Shape& shape, const std::shared_ptr<NumberType>& type)
			    : m_data(nullptr), m_shape(shape), m_type(type)
			{
				// shape is known at build time, let's preallocate it
				// we could inforce immutability at this level somehow?
				if (shape.is_static())
					m_data = std::make_unique<MemoryManager>(
					    get_size_from_shape(m_shape), type);
			}

			static std::shared_ptr<Storage> create_view(
			    void* ptr, const Shape& shape,
			    const std::shared_ptr<NumberType>& type)
			{
				return std::make_shared<Storage>(ptr, shape, type, View{});
			}

			/* Initialiser called by device_put.
			 */
			Storage(
			    void* ptr, const Shape& shape,
			    const std::shared_ptr<NumberType>& type, const bool copy, Copy)
			    : m_shape(shape), m_type(type)
			{
				if (copy)
					m_data = MemoryManager::copy_from(
					    ptr, get_size_from_shape(shape), type);
				else
					m_data = MemoryManager::move_from(
					    ptr, get_size_from_shape(shape), type);
			}

			Storage(
			    void* ptr, const Shape& shape,
			    const std::shared_ptr<NumberType>& type, View)
			    : m_shape(shape), m_type(type)
			{
				m_data =
				    MemoryManager::view(ptr, get_size_from_shape(shape), type);
			}

			void* get_copy() const
			{
				return m_data->get_copy(get_size_from_shape(m_shape), m_type);
			}

			/* Allocates memory storage on the fly with a given shape.
			 * If the exact is not static, e.g. the exact dimensions are
			 * not known, at runtime, this will throw.
			 *
			 */
			void allocate_storage(const Shape& shape)
			{
				if (!shape.is_static())
					error(
					    "Cannot allocate tensor with shape {}, with "
					    "unknown size requirements",
					    shape.to_string());
				// new allocation, call allocator_dispatch
				if (!m_data)
				{
					set_shape(shape);
					m_data = std::make_unique<MemoryManager>(
					    get_size_from_shape(shape), m_type);
				}
				else
				{
					// memory already allocated, and we have the right shape
					// nothing to do
					if (m_shape == shape)
						return;
					// memory has been allocated, but it isn't the same
					// shape
					else
					{
						auto old_shape = m_shape;
						set_shape(shape);
						// if the size requirement is larger, reallocate
						// memory
						if (get_size_from_shape(shape) > size())
						{
							m_data->realloc(size(), get_size_from_shape(shape), m_type);
						}
						// otherwise nothing happens, we just own a larger
						// memory block but only use part of it
					}
				}
			}

			[[nodiscard]] const Shape& get_shape() const noexcept
			{
				return m_shape;
			}

			[[nodiscard]] const std::shared_ptr<NumberType>& get_type() const
			    noexcept
			{
				return m_type;
			}

			[[nodiscard]] void* data() { return m_data->m_internal_data.get(); }

			[[nodiscard]] size_t size() const
			{
				return get_size_from_shape(m_shape);
			}

		protected:
			/* Sets the static Shape. There is a guarantee that
			 * the Shape passed to this function is static.
			 */
			void set_static_shape(const Shape& shape)
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

			/* Sets the runtime Shape when static Shape allocation
			 * was not possible, e.g. the user provided Dynamic
			 * shapes in the Graph construction. If Storage already
			 * owns data the checks are skipped.
			 */
			void set_shape(const Shape& shape)
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

			/* Sets the runtime Shape when static Shape allocation
			 * was not possible, e.g. the user provided Dynamic
			 * shapes in the Graph construction. If Storage already
			 * owns data the checks are skipped.
			 */
			void reshape(const Shape& shape)
			{
				if (!shape.is_static())
					error("Cannot set dynamic shape in storage.");
				if (size() != get_size_from_shape(shape))
				{
					error(
					    "Cannot modify the total size of shape: {} vs {}",
					    m_shape.to_string(), shape.to_string());
				}
				m_shape = shape;
			}

		private:
			[[nodiscard]] static size_t get_size_from_shape(const Shape& size)
			{
				return std::accumulate(
				    size.begin(), size.end(), size_t{1}, std::multiplies{});
			}

		protected:
			std::unique_ptr<MemoryManager> m_data;
			Shape m_shape;
			std::shared_ptr<NumberType> m_type;
		};
	} // namespace graph
} // namespace shogun

#endif
