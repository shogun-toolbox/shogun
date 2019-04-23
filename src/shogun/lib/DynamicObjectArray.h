/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Evgeniy Andreev,
 *          Sergey Lisitsyn, Leon Kuchenbecker, Yuyu Zhang, Thoralf Klein,
 *          Fernando Iglesias, Bjoern Esser, Viktor Gal
 */

#ifndef _DYNAMIC_OBJECT_ARRAY_H_
#define _DYNAMIC_OBJECT_ARRAY_H_

#include <algorithm>
#include <type_traits>
#include <random>
#include <vector>

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>
#include <shogun/io/Serializable.h>
#include <shogun/util/converters.h>
#include <shogun/mathematics/RandomNamespace.h>

namespace shogun
{
/** @brief Dynamic array class for SGObject pointers that creates an array
 * that can be used like a list or an array.
 *
 * It grows and shrinks dynamically, while elements can be accessed
 * via index. It only stores SGObject pointers, which ARE automagically
 * SG_REF'd/deleted.
 *
 */
class DynamicObjectArray : public SGObject
{
	public:
		/** default constructor */
		DynamicObjectArray()
		: SGObject(), m_array()
		{
			dim1_size = 1;
			dim2_size = 1;
			init();
		}

		/** constructor
		 *
		 * @param dim1 dimension 1
		 * @param dim2 dimension 2
		 */
		DynamicObjectArray(size_t dim1, size_t dim2 = 1)
		: SGObject()
		{
			dim1_size = dim1;
			dim2_size = dim2;
			m_array.reserve(dim1_size*dim2_size);
			init();
		}

		DynamicObjectArray(std::shared_ptr<SGObject>* p_array, size_t dim1, size_t dim2, bool p_free_array=true, bool p_copy_array=false)
		: SGObject(), m_array(p_array, p_array + dim1*dim2)
		{
			dim1_size = dim1;
			dim2_size = dim2;

			init();
		}

		virtual ~DynamicObjectArray() { }

		/** get array size (including granularity buffer)
		 *
		 * @return total array size (including granularity buffer)
		 */
		inline index_t get_array_size() const
		{
			return utils::safe_convert<index_t>(m_array.capacity());
		}

		/** get number of elements
		 *
		 * @return number of elements
		 */
		inline index_t get_num_elements() const
		{
			return utils::safe_convert<index_t>(m_array.size());
		}

		/** get array element at index
		 *
		 * (does NOT do bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline std::shared_ptr<SGObject> get_element(size_t index) const
		{
			return m_array[index];
		}

		template<class T>
		inline std::shared_ptr<T> get_element(size_t index) const
		{
			return std::dynamic_pointer_cast<T>(m_array[index]);
		}

		/** get array element at index
		 *
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline std::shared_ptr<SGObject> element(int32_t idx1, int32_t idx2=0, int32_t idx3=0)
		{
			return get_element(idx1+dim1_size*(idx2+dim2_size*idx3));
		}

		/** get last array element
		 *
		 * @return last array element
		 */
		inline std::shared_ptr<SGObject> get_last_element() const
		{
			return m_array.back();
		}

		/** get last array element
		 *
		 * @return last array element
		 */
		template<class T>
		inline std::shared_ptr<T> get_last_element() const
		{
			return std::dynamic_pointer_cast<T>(m_array.back());
		}

		/** get array element at index
		 *
		 * (does bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline std::shared_ptr<SGObject> get_element_safe(int32_t index) const
		{
			if (index >= utils::safe_convert<index_t>(m_array.size()))
			{
				error("array index out of bounds ({} >= {})",
						 index, m_array.size());
			}
			return get_element(index);
		}

		template<class T>
		inline std::shared_ptr<T> get_element_safe(int32_t index) const
		{
			return std::dynamic_pointer_cast<T>(get_element_safe(index));
		}

#ifndef SWIG
		SG_FORCED_INLINE std::shared_ptr<SGObject> at(int32_t index) const
		{
			return get_element_safe(index);
		}
#endif

		/** set array element at index
		 *
		 * @param e element to set
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 2
		 * @return if setting was successful
		 */
		inline bool set_element(std::shared_ptr<SGObject> e, size_t index)
		{
			if (index >= m_array.size())
				m_array.resize(index);
			m_array[index] = e;
			return true;
		}

		/** insert array element at index
		 *
		 * @param e element to insert
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool insert_element(std::shared_ptr<SGObject> e, int32_t index)
		{
			m_array.insert(m_array.begin()+index, e);
			return true;
		}

		template <typename T, typename T2 = typename std::enable_if_t<std::is_arithmetic<T>::value>>
		inline bool append_element(T e, const char* name="")
		{
			return append_element(std::make_shared<Serializable<T>>(e, name));
		}

		template <typename T>
		inline bool append_element(SGVector<T> e, const char* name="")
		{
			return append_element(std::make_shared<VectorSerializable<T>>(e, name));
		}

		template <typename T>
		inline bool append_element(SGMatrix<T> e, const char* name="")
		{
			return append_element(std::make_shared<MatrixSerializable<T>>(e, name));
		}

		template <typename T>
		inline bool append_element(const std::vector<SGVector<T>>& e, const char* name="")
		{
			return append_element(std::make_shared<VectorListSerializable<T>>(e, name));
		}

		/** append array element to the end of array
		 *
		 * @param e element to append
		 * @return if setting was successful
		 */
		inline bool append_element(std::shared_ptr<SGObject> e)
		{
			m_array.push_back(e);
			return true;
		}

		/** STD VECTOR compatible. Append array element to the end
		 *  of array.
		 *
		 * @param e element to append
		 */
		inline void push_back(std::shared_ptr<SGObject> e)
		{
			m_array.push_back(e);
		}

		/** STD VECTOR compatible. Delete array element at the end
		 *  of array.
		 */
		inline void pop_back()
		{
			m_array.pop_back();
		}

		/** STD VECTOR compatible. Return array element at the end
		 *  of array.
		 *
		 * @return element at the end of array
		 */
		inline std::shared_ptr<SGObject> back() const
		{
			return m_array.back();
		}

		/** find first occurence of array element and return its index
		 * or -1 if not available
		 *
		 * @param elem element to search for
		 * @return index of element or -1
		 */
		inline int32_t find_element(std::shared_ptr<SGObject> elem) const
		{
			auto it = std::find(m_array.begin(), m_array.end(), elem);
			if (it != m_array.end())
				return std::distance(m_array.begin(), it);
			return -1L;
		}

		/** delete array element at idx
		 * (does not call SG_FREE() or the like)
		 *
		 * @param idx index
		 * @return if deleting was successful
		 */
		inline bool delete_element(size_t idx)
		{
			auto e=m_array[idx];
			m_array.erase(std::remove(m_array.begin(), m_array.end(), e),
				m_array.end());
			return true;
		}

		inline void clear_array()
		{
			m_array.assign(m_array.size(), nullptr);
		}

		/** resets the array */
		inline void reset_array()
		{
			m_array.clear();
		}

		/** operator overload for array assignment
		 *
		 * @param orig original array
		 * @return new array
		 */
		inline DynamicObjectArray& operator=(DynamicObjectArray& orig)
		{
			/* copy pointer DynArray */
			m_array=orig.m_array;
			return *this;
		}

		/** @return underlying array of pointers */
		inline std::shared_ptr<SGObject>* get_array() { return m_array.data(); }

#ifndef SWIG // SWIG should skip this part
		inline auto begin()
		{
			return m_array.begin();
		}

		inline auto end()
		{
			return m_array.end();
		}
#endif // SWIG

		/** @return object name */
		virtual const char* get_name() const
		{ return "DynamicObjectArray"; }

		/** Can (optionally) be overridden to pre-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_PRE
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void save_serializable_pre() noexcept(false)
		{
			SGObject::save_serializable_pre();
			m_array.shrink_to_fit();
		}

	private:
		/** register parameters */
		virtual void init()
		{
			watch_param("array", &m_array);

			SG_ADD(&dim1_size, "dim1_size", "Dimension 1");
			SG_ADD(&dim2_size, "dim2_size", "Dimension 2");
		}

	private:
		/** underlying array */
		std::vector<std::shared_ptr<SGObject>> m_array;

		/** dimension 1 */
		index_t dim1_size;

		/** dimension 2 */
		index_t dim2_size;
};
}
#endif /* _DYNAMIC_OBJECT_ARRAY_H_  */
