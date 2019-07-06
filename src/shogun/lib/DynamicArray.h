/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Heiko Strathmann,
 *          Sergey Lisitsyn, Leon Kuchenbecker, Yuyu Zhang, Thoralf Klein,
 *          Fernando Iglesias, Bjoern Esser
 */

#ifndef _DYNAMIC_ARRAY_H_
#define _DYNAMIC_ARRAY_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>

#include <random>
#include <vector>

namespace shogun
{
/** @brief Template Dynamic array class that creates an array that can
 * be used like a list or an array.
 *
 * It grows and shrinks dynamically, while elements can be accessed
 * via index.  It is performance tuned for simple types like float
 * etc. and for hi-level objects only stores pointers, which are not
 * automagically SG_REF'd/deleted.
 */
template <class T> class DynamicArray :public SGObject
{
		using DiffType = typename std::vector<T>::difference_type;
		using ReferenceType = typename std::vector<T>::reference;
		using ConstReferenceType = typename std::vector<T>::const_reference;
	public:
		/** default constructor */
		DynamicArray()
		: SGObject(), m_array()
		{
			dim1_size=1;
			dim2_size=1;
			dim3_size=1;

			init();
		}

		/** constructor
		 *
		 * @param p_dim1_size dimension 1
		 * @param p_dim2_size dimension 2
		 * @param p_dim3_size dimension 3
		 */
		DynamicArray(int32_t p_dim1_size, int32_t p_dim2_size=1, int32_t p_dim3_size=1)
		: SGObject(), m_array(p_dim1_size*p_dim2_size*p_dim3_size)
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=p_dim3_size;

			init();
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		DynamicArray(T* p_array, int32_t p_dim1_size, bool p_free_array, bool p_copy_array)
		: SGObject()
		{
			dim1_size=p_dim1_size;
			dim2_size=1;
			dim3_size=1;

			set_array(p_array, p_dim1_size, p_free_array, p_copy_array);
			init();
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_dim2_size dimension 2
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		DynamicArray(T* p_array, int32_t p_dim1_size, int32_t p_dim2_size,
						bool p_free_array, bool p_copy_array)
		: SGObject()
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=1;

			set_array(p_array, p_dim1_size, p_dim2_size, p_free_array, p_copy_array);
			init();
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_dim2_size dimension 2
		 * @param p_dim3_size dimension 3
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		DynamicArray(T* p_array, int32_t p_dim1_size, int32_t p_dim2_size,
						int32_t p_dim3_size, bool p_free_array, bool p_copy_array)
		: SGObject()
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=p_dim3_size;

			set_array(p_array, p_dim1_size, p_dim2_size, p_dim3_size, p_free_array, p_copy_array);

			init();
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_dim2_size dimension 2
		 * @param p_dim3_size dimension 3
		 */
		DynamicArray(const T* p_array, int32_t p_dim1_size=1, int32_t p_dim2_size=1, int32_t p_dim3_size=1)
		: SGObject(), m_array(p_array, p_array+(p_dim1_size*p_dim2_size*p_dim3_size))
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=p_dim3_size;

			init();
		}

		virtual ~DynamicArray() { m_array.clear(); }

		/** get array size (including granularity buffer)
		 *
		 * @return total array size
		 */
		inline int32_t get_array_size()
		{
			return m_array.capacity();
		}

		/** return 2d array size
		 *
		 * @param dim1 dimension 1 will be stored here
		 * @param dim2 dimension 2 will be stored here
		 */
		inline void get_array_size(int32_t& dim1, int32_t& dim2)
		{
			dim1=dim1_size;
			dim2=dim2_size;
		}

		/** return 3d array size
		 *
		 * @param dim1 dimension 1 will be stored here
		 * @param dim2 dimension 2 will be stored here
		 * @param dim3 dimension 3 will be stored here
		 */
		inline void get_array_size(int32_t& dim1, int32_t& dim2, int32_t& dim3)
		{
			dim1=dim1_size;
			dim2=dim2_size;
			dim3=dim3_size;
		}

		/** get dimension 1
		 *
		 * @return dimension 1
		 */
		inline int32_t get_dim1() { return dim1_size; }

		/** get dimension 2
		 *
		 * @return dimension 2
		 */
		inline int32_t get_dim2() { return dim2_size; }

		/** get dimension 3
		 *
		 * @return dimension 3
		 */
		inline int32_t get_dim3() { return dim3_size; }

		/** get number of elements
		 *
		 * @return number of elements
		 */
		inline int32_t get_num_elements() const
		{
			return m_array.size();
		}

		/** get array element at index
		 *
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline ConstReferenceType get_element(int32_t idx1, int32_t idx2=0, int32_t idx3=0) const
		{
			return m_array[idx1+dim1_size*(idx2+dim2_size*idx3)];
		}

		/** get array element at index
		 *
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline ConstReferenceType element(int32_t idx1, int32_t idx2=0, int32_t idx3=0) const
		{
			return get_element(idx1, idx2, idx3);
		}

		/** get array element at index
		 *
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline ReferenceType element(int32_t idx1, int32_t idx2=0, int32_t idx3=0)
		{
			return m_array[idx1+dim1_size*(idx2+dim2_size*idx3)];
		}

		/** get element of given array at given index
		 *
		 * @param p_array another array
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline T& element(T* p_array, int32_t idx1, int32_t idx2=0, int32_t idx3=0)
		{
			ASSERT(idx1>=0 && idx1<dim1_size)
			ASSERT(idx2>=0 && idx2<dim2_size)
			ASSERT(idx3>=0 && idx3<dim3_size)
			return p_array[idx1+dim1_size*(idx2+dim2_size*idx3)];
		}

		/** get element of given array at given index
		 *
		 * @param p_array another array
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @param p_dim1_size size of dimension 1
		 * @param p_dim2_size size of dimension 2
		 * @return element of given array at given index
		 */
		inline T& element(T* p_array, int32_t idx1, int32_t idx2, int32_t idx3, int32_t p_dim1_size, int32_t p_dim2_size)
		{
			ASSERT(p_dim1_size==dim1_size)
			ASSERT(p_dim2_size==dim2_size)
			ASSERT(idx1>=0 && idx1<p_dim1_size)
			ASSERT(idx2>=0 && idx2<p_dim2_size)
			ASSERT(idx3>=0 && idx3<dim3_size)
			return p_array[idx1+p_dim1_size*(idx2+p_dim2_size*idx3)];
		}

		/** gets last array element
		 *
		 * @return array element at last index
		 */
		inline ReferenceType get_last_element() const
		{
			return m_array.back();
		}

		/** get array element at index
		 *
		 * (does bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline ReferenceType get_element_safe(int32_t index) const
		{
			if (index>=get_num_elements())
			{
				SG_SERROR("array index out of bounds (%d >= %d)\n",
						index, get_num_elements());
			}
			return m_array[index];
		}

		/** set array element at index
		 *
		 * @param e element to set
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 2
		 * @return if setting was successful
		 */
		inline bool set_element(T e, int32_t idx1, int32_t idx2=0, int32_t idx3=0)
		{
			auto index = idx1+dim1_size*(idx2+dim2_size*idx3);
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
		inline bool insert_element(T e, int32_t index)
		{
			m_array.insert(m_array.begin()+index, e);
			return true;
		}

		/** append array element to the end of array
		 *
		 * @param e element to append
		 * @return if setting was successful
		 */
		inline bool append_element(T e)
		{
			m_array.push_back(e);
			return true;
		}

		/** STD VECTOR compatible. Append array element to the end
		 *  of array.
		 *
		 * @param e element to append
		 */
		inline void push_back(T e)
		{ m_array.push_back(e); }

		/** STD VECTOR compatible. Delete array element at the end
		 *  of array.
		 */
		inline void pop_back()
		{
			m_array.pop_back();
		}

		/** STD  VECTOR compatible. Return array element at the end
		 *  of array.
		 *
		 * @return element at the end of array
		 */
		inline T back()
		{
			return m_array.back();
		}

		/** find first occurence of array element and return its index
		 * or -1 if not available
		 *
		 * @param e element to search for
		 * @return index of element or -1
		 */
		inline int32_t find_element(T e)
		{
			auto it = std::find(m_array.begin(), m_array.end(), e);
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
		inline bool delete_element(int32_t idx)
		{
                        auto e=m_array[idx];
                        m_array.erase(std::remove(m_array.begin(), m_array.end(), e),
		                                m_array.end());
                        return true;			
		}

		/** resize array
		 *
		 * @param ndim1 new dimension 1
		 * @param ndim2 new dimension 2
		 * @param ndim3 new dimension 3
		 * @return if resizing was successful
		 */
		inline bool resize_array(int32_t ndim1, int32_t ndim2=1, int32_t ndim3=1)
		{
			dim1_size=ndim1;
			dim2_size=ndim2;
			dim3_size=ndim3;
			try
			{
				m_array.reserve(ndim1*ndim2*ndim3);
				return true;
			}
			catch (std::exception& e)
			{
				return false;
			}
		}

		/** set array with a constant */
		void set_const(const T& const_element)
		{
			m_array.assign(m_array.capacity(), const_element);
		}

		/** get the array
		 * call get_array just before messing with it DO NOT call any
		 * [],resize/delete functions after get_array(), the pointer may
		 * become invalid !
		 *
		 * @return the array
		 */
		inline T* get_array()
		{
			return m_array.data();
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array new array
		 * @param p_num_elements last element index + 1
		 * @param array_size number of elements in array
		 */
		inline void set_array(T* p_array, int32_t p_num_elements,
							  int32_t array_size)
		{
			m_array.resize(array_size);
			std::copy(p_array, p_array+p_num_elements, m_array.begin());
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param dim1 dimension 1
		 * @param p_free_array if array must be freed
		 * @param copy_array if array must be copied
		 */
		inline void set_array(T* p_array, int32_t dim1,
						bool p_free_array, bool copy_array)
		{
			if (!copy_array)
				throw std::runtime_error("not copying array is not supported");
			dim1_size=dim1;
			dim2_size=1;
			dim3_size=1;

			m_array.resize(dim1);
			std::copy(p_array, p_array+dim1, m_array.begin());
		}

		/** set the 2d array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param dim1 dimension 1
		 * @param dim2 dimension 2
		 * @param p_free_array if array must be freed
		 * @param copy_array if array must be copied
		 */
		inline void set_array(T* p_array, int32_t dim1,
						int32_t dim2, bool p_free_array, bool copy_array)
		{
			if (!copy_array)
				throw std::runtime_error("not copying array is not supported");
			dim1_size=dim1;
			dim2_size=dim2;
			dim3_size=1;

			m_array.reserve(dim1+dim2);
			std::copy(p_array, p_array+dim1+dim2, m_array.begin());
		}

		/** set the 3d array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param dim1 dimension 1
		 * @param dim2 dimension 2
		 * @param dim3 dimension 3
		 * @param p_free_array if array must be freed
		 * @param copy_array if array must be copied
		 */
		inline void set_array(T* p_array, int32_t dim1,
						int32_t dim2, int32_t dim3, bool p_free_array, bool copy_array)
		{
			if (!copy_array)
				throw std::runtime_error("not copying array is not supported");
			dim1_size=dim1;
			dim2_size=dim2;
			dim3_size=dim3;

			m_array.clear();
			m_array.reserve(dim1+dim2+dim3);
			std::copy(p_array, p_array+dim1+dim2+dim3, m_array.begin());
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param p_size size of another array
		 */
		inline void set_array(const T* p_array, int32_t p_size)
		{
			m_array.resize(p_size);
			std::copy(p_array, p_array+p_size, m_array.begin());
		}

		/** clear the array (with e.g. zeros)
		 * @param value value to fill array with
		 */
		inline void clear_array(T value)
		{
			m_array.assign(m_array.capacity(), value);
		}

		/** resets the array */
		inline void reset_array()
		{
			m_array.clear();
		}

		/** operator overload for array read only access
		 * use set_element() for write access (will also make the array
		 * dynamically grow)
		 *
		 * DOES NOT DO ANY BOUNDS CHECKING
		 *
		 * @param index index
		 * @return element at index
		 */
		inline ConstReferenceType operator[](int32_t index) const
		{
			return m_array[index];
		}

		/** operator overload for array read-write access
		 *
		 * DOES NOT DO ANY BOUNDS CHECKING
		 *
		 * @param index index
		 * @return element at index
		 */
		inline ReferenceType operator[](int32_t index)
		{
			return m_array[index];
		}

		/** operator overload for array assignment
		 *
		 * @param orig original array
		 * @return new array
		 */
		inline DynamicArray<T>& operator=(DynamicArray<T>& orig)
		{
			m_array=orig.m_array;
			dim1_size=orig.dim1_size;
			dim2_size=orig.dim2_size;
			dim3_size=orig.dim3_size;

			return *this;
		}

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

		/** display this array */
		inline void display_array()
		{
			if (get_name())
				io::print("DynamicArray '{}' of size: {}x{}x{}\n", get_name(), dim1_size, dim2_size, dim3_size);
			else
				io::print("DynamicArray of size: {}x{}x{}\n",dim1_size, dim2_size, dim3_size);

			for (int32_t k=0; k<dim3_size; k++)
				for (int32_t i=0; i<dim1_size; i++)
				{
					io::print("element({},:,{}) = [ ",i, k);
					for (int32_t j=0; j<dim2_size; j++)
						io::print("{:1.1f},", (float32_t) element(i,j,k));
					io::print(" ]\n");
				}
		}

		/** display array's size */
		inline void display_size()
		{
			io::print("DynamicArray of size: {}x{}x{}\n",dim1_size, dim2_size, dim3_size);
		}

		/** @return object name */
		virtual const char* get_name() const
		{
			return "DynamicArray";
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
			SGObject::save_serializable_pre();
			m_array.shrink_to_fit();
		}

		virtual std::shared_ptr<SGObject> clone(ParameterProperties pp) const override
		{
			return SGObject::clone(pp)->template as<DynamicArray>();
		}

	private:

		/** register parameters */
		virtual void init()
		{
			set_generic<T>();
			watch_param("array", &m_array);
			SG_ADD(&dim1_size, "dim1_size", "Dimension 1");
			SG_ADD(&dim2_size, "dim2_size", "Dimension 2");
			SG_ADD(&dim3_size, "dim3_size", "Dimension 3");
		}

	protected:

		/** underlying array */
		std::vector<T> m_array;

		/** dimension 1 */
		int32_t dim1_size;

		/** dimension 2 */
		int32_t dim2_size;

		/** dimension 3 */
		int32_t dim3_size;
};
}
#endif /* _DYNAMIC_ARRAY_H_  */
