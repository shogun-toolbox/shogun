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
#include <shogun/base/DynArray.h>
#include <shogun/base/Parameter.h>
#include <shogun/io/Serializable.h>
#include <shogun/util/converters.h>



namespace shogun
{
/** @brief Dynamic array class for CSGObject pointers that creates an array
 * that can be used like a list or an array.
 *
 * It grows and shrinks dynamically, while elements can be accessed
 * via index. It only stores CSGObject pointers, which ARE automagically
 * SG_REF'd/deleted.
 *
 */
class CDynamicObjectArray : public CSGObject
{
	public:
		/** default constructor */
		CDynamicObjectArray()
		: CSGObject(), m_array()
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
		CDynamicObjectArray(size_t dim1, size_t dim2 = 1)
		: CSGObject()
		{
			dim1_size = dim1;
			dim2_size = dim2;
			m_array.reserve(dim1_size*dim2_size);
			init();
		}

		CDynamicObjectArray(CSGObject** p_array, size_t dim1, size_t dim2, bool p_free_array=true, bool p_copy_array=false)
		: CSGObject(), m_array(dim1*dim2)
		{
			m_array.assign(p_array, p_array+(dim1*dim2));
			dim1_size = dim1;
			dim2_size = dim2;

			init();
		}

		virtual ~CDynamicObjectArray() { unref_all(); }

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
		inline CSGObject* get_element(size_t index) const
		{
			auto elem = m_array[index];
			SG_REF(elem);
			return elem;
		}

		/** get array element at index
		*
		* @param idx1 index 1
		* @param idx2 index 2
		* @param idx3 index 3
		* @return array element at index
		*/
		inline CSGObject* element(int32_t idx1, int32_t idx2=0, int32_t idx3=0) const
		{
			return get_element(idx1+dim1_size*(idx2+dim2_size*idx3));
		}

		/** get last array element
		 *
		 * @return last array element
		 */
		inline CSGObject* get_last_element() const
		{
			auto e = m_array.back();
			SG_REF(e);
			return e;
		}

		/** get array element at index
		 *
		 * (does bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline CSGObject* get_element_safe(size_t index) const
		{
			if (index >= m_array.size())
			{
				SG_SERROR("array index out of bounds (%d >= %d)\n",
						 index, m_array.size());
			}
			return get_element(index);
		}

		inline CSGObject* at(int32_t index) const
		{
			return get_element_safe(index);
		}

		/** set array element at index
		 *
		 * @param e element to set
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 2
		 * @return if setting was successful
		 */
		inline bool set_element(CSGObject* e, size_t index)
		{
			CSGObject* old = nullptr;
			if (index < m_array.size())
				old = m_array[index];
			else
				m_array.resize(index);

			SG_REF(e);
			m_array[index] = e;
			SG_UNREF(old);
			return true;
		}

		/** insert array element at index
		 *
		 * @param e element to insert
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool insert_element(CSGObject* e, size_t index)
		{
			SG_REF(e);
			m_array.insert(m_array.begin()+index, e);
			return true;
		}

		template <typename T, typename T2 = typename std::enable_if_t<std::is_arithmetic<T>::value>>
		inline bool append_element(T e, const char* name="")
		{
			auto serialized_element = new CSerializable<T>(e, name);
			return append_element(serialized_element);
		}

		template <typename T>
		inline bool append_element(SGVector<T> e, const char* name="")
		{
			auto serialized_element = new CVectorSerializable<T>(e, name);
			return append_element(serialized_element);
		}

		template <typename T>
		inline bool append_element(SGMatrix<T> e, const char* name="")
		{
			auto serialized_element = new CMatrixSerializable<T>(e, name);
			return append_element(serialized_element);
		}

		template <typename T>
		inline bool append_element(SGStringList<T> e, const char* name="")
		{
			auto serialized_element = new CStringListSerializable<T>(e, name);
			return append_element(serialized_element);
		}

		/** append array element to the end of array
		 *
		 * @param e element to append
		 * @return if setting was successful
		 */
		inline bool append_element(CSGObject* e)
		{
			m_array.push_back(e);
			SG_REF(e);
			return true;
		}

		/** STD VECTOR compatible. Append array element to the end
		 *  of array.
		 *
		 * @param e element to append
		 */
		inline void push_back(CSGObject* e)
		{
			m_array.push_back(e);
			SG_REF(e);
		}

		/** STD VECTOR compatible. Delete array element at the end
		 *  of array.
		 */
		inline void pop_back()
		{
			auto e = m_array.back();
			SG_UNREF(e);

			m_array.pop_back();
		}

		/** STD VECTOR compatible. Return array element at the end
		 *  of array.
		 *
		 * @return element at the end of array
		 */
		inline CSGObject* back() const
		{
			auto e=m_array.back();
			SG_REF(e);
			return e;
		}

		/** find first occurence of array element and return its index
		 * or -1 if not available
		 *
		 * @param elem element to search for
		 * @return index of element or -1
		 */
		inline index_t find_element(CSGObject* elem) const
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
			SG_UNREF(e);
			return true;
		}

		inline void clear_array()
		{
			unref_all();
			m_array.assign(m_array.size(), nullptr);
		}

		/** resets the array */
		inline void reset_array()
		{
			unref_all();
			m_array.clear();
		}

		/** operator overload for array assignment
		 *
		 * @param orig original array
		 * @return new array
		 */
		inline CDynamicObjectArray& operator=(CDynamicObjectArray& orig)
		{
			/* SG_REF all new elements (implicitly) */
			for (auto& v: orig.m_array)
				SG_REF(v);

			/* unref after adding to avoid possible deletion */
			unref_all();

			/* copy pointer DynArray */
			m_array=orig.m_array;
			return *this;
		}

		/** @return underlying array of pointers */
		inline CSGObject** get_array() { return m_array.data(); }

		/** shuffles the array (not thread safe!) */
		inline void shuffle()
		{
			auto rng = std::default_random_engine {};
			std::shuffle(m_array.begin(), m_array.end(), rng); 
		}

		/** shuffles the array with external random state */
		inline void shuffle(CRandom * rand)
		{
			using DiffType = decltype(m_array)::difference_type;
			auto first = m_array.begin();
			auto last = m_array.end();
			DiffType n = last - first;
			for (DiffType i = n-1; i > 0; --i)
			{
				using std::swap;
				swap(first[i], first[rand->random(DiffType(0), i)]);
			}
		}

		/** @return object name */
		virtual const char* get_name() const
		{ return "DynamicObjectArray"; }

		// without this definition R interface is missing these inherited functions
		using CSGObject::save_serializable;
		using CSGObject::load_serializable;

		/** Can (optionally) be overridden to pre-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_PRE
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void load_serializable_pre() noexcept(false)
		{
			CSGObject::load_serializable_pre();

			m_array.resize_array(m_array.get_num_elements(), true);
		}

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
			CSGObject::save_serializable_pre();
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

		/** de-reference all elements of this array once */
		inline void unref_all()
		{
			/* SG_UNREF all my elements */
			for (auto& o: m_array)
				SG_UNREF(o);
		}

	private:
		/** underlying array */
		std::vector<CSGObject*> m_array;

		/** dimension 1 */
		index_t dim1_size;

		/** dimension 2 */
		index_t dim2_size;
};
}
#endif /* _DYNAMIC_OBJECT_ARRAY_H_  */
