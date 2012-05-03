/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
 */
#ifndef __SGVECTOR_H__
#define __SGVECTOR_H__

namespace shogun
{
/** @brief shogun vector */
template<class T> class SGVector
{
	public:
		/** default constructor */
		SGVector() : vector(NULL), vlen(0), m_refcount(NULL) { }

		/** constructor for setting params */
		SGVector(T* v, index_t len, bool ref_counting=true)
			: vector(v), vlen(len), m_refcount(NULL)
		{
			if (ref_counting)
				m_refcount=SG_CALLOC(int32_t, 1); 

			ref();
		}

		/** constructor to create new vector in memory */
		SGVector(index_t len, bool ref_counting=true)
			: vlen(len), m_refcount(NULL)
		{
			if (ref_counting)
				m_refcount=SG_CALLOC(int32_t, 1);

			vector=SG_MALLOC(T, len);

			ref();
		}

		/** copy constructor */
		SGVector(const SGVector &orig)
			: vector(orig.vector), vlen(orig.vlen), m_refcount(orig.m_refcount) 
		{
			ref();
		}

		/** empty destructor */
		virtual ~SGVector()
		{
			unref();
		}

#ifdef USE_REFERENCE_COUNTING
		/** increase reference counter
		 *
		 * @return reference count
		 */
		int32_t ref()
		{
			if (m_refcount == NULL)
			{
				return -1;
			}

			++(*m_refcount);
			return *m_refcount;
		}

		/** display reference counter
		 *
		 * @return reference count
		 */
		int32_t ref_count()
		{
			if (m_refcount == NULL)
				return -1;

			return *m_refcount;
		}

		/** decrement reference counter and deallocate object if refcount is zero
		 * before or after decrementing it
		 *
		 * @return reference count
		 */
		int32_t unref()
		{
			if (m_refcount == NULL)
				return -1;

			if (*m_refcount==0 || --(*m_refcount)==0)
			{
				SG_FREE(vector);
				SG_FREE(m_refcount);

				vector=NULL;
				m_refcount=NULL;
				vlen=0;

				return 0;
			}
			else
			{
				return *m_refcount;
			}
		}

#endif //USE_REFERENCE_COUNTING

		/** fill vector with zeros */
		void zero()
		{
			if (vector && vlen)
				set_const(0);
		}

		/** set vector to a constant */
		void set_const(T const_elem)
		{
			for (index_t i=0; i<vlen; i++)
				vector[i]=const_elem ;
		}

		/** range fill */
		void range_fill(T start=0)
		{
			range_fill_vector(vector, vlen, start);
		}

		/** random */
		void random(T min_value, T max_value)
		{
			random_vector(vector, vlen, min_value, max_value);
		}

		/** random permutate */
		void randperm()
		{
			/* this does not work. Heiko Strathmann */
			SG_SNOTIMPLEMENTED;
			randperm(vector, vlen);
		}

		/** clone vector */
		SGVector<T> clone()
		{
			SGVector<T> c;
			c.vector=clone_vector(vector, vlen);
			c.vlen=vlen;
			//c.do_free=true;

			return c;
		}

		/** clone vector */
		template <class VT>
		static VT* clone_vector(const VT* vec, int32_t len)
		{
			VT* result = SG_MALLOC(VT, len);
			for (int32_t i=0; i<len; i++)
				result[i]=vec[i];

			return result;
		}

		/** fill vector */
		template <class VT>
		static void fill_vector(VT* vec, int32_t len, VT value)
		{
			for (int32_t i=0; i<len; i++)
				vec[i]=value;
		}

		/** range fill vector */
		template <class VT>
		static void range_fill_vector(VT* vec, int32_t len, VT start=0)
		{
			for (int32_t i=0; i<len; i++)
				vec[i]=i+start;
		}

		/** random vector */
		template <class VT>
		static void random_vector(VT* vec, int32_t len, VT min_value, VT max_value)
		{
			//FIXME for (int32_t i=0; i<len; i++)
			//FIXME 	vec[i]=CMath::random(min_value, max_value);
		}

		/** random permatutaion */
		template <class VT>
		static void randperm(VT* perm, int32_t n)
		{
			for (int32_t i = 0; i < n; i++)
				perm[i] = i;
			permute(perm,n);
		}

		/** permute */
		template <class VT>
		static void permute(VT* perm, int32_t n)
		{
			//FIXME for (int32_t i = 0; i < n; i++)
			//FIXME 	CMath::swap(perm[random(0, n - 1)], perm[i]);
		}

		/** get vector element at index
		 *
		 * @param index index
		 * @return vector element at index
		 */
		const T& get_element(index_t index)
		{
			ASSERT(vector && (index>=0) && (index<vlen));
			return vector[index];
		}

		/** set vector element at index 'index' return false in case of trouble
		 *
		 * @param p_element vector element to set
		 * @param index index
		 * @return if setting was successful
		 */
		void set_element(const T& p_element, index_t index)
		{
			ASSERT(vector && (index>=0) && (index<vlen));
			vector[index]=p_element;
		}

		/** resize vector
		 *
		 * @param n new size
		 * @return if resizing was successful
		 */
		void resize_vector(int32_t n)
		{
			vector=SG_REALLOC(T, vector, n);

			if (n > vlen)
				memset(&vector[vlen], 0, (n-vlen)*sizeof(T));
			vlen=n;
		}

		/** operator overload for vector read only access
		 *
		 * @param index dimension to access
		 *
		 */
		inline const T& operator[](index_t index) const
		{
			return vector[index];
		}

		/** operator overload for vector r/w access
		 *
		 * @param index dimension to access
		 *
		 */
		inline T& operator[](index_t index)
		{
			return vector[index];
		}

		/** display array size */
		void display_size() const
		{
			SG_SPRINT("SGVector '%p' of size: %d\n", vector, vlen);
		}

		/** display array */
		void display_vector() const
		{
			display_size();
			for (int32_t i=0; i<vlen; i++)
				SG_SPRINT("%10.10g,", (float64_t) vector[i]);
			SG_SPRINT("\n");
		}

	public:
		/** vector  */
		T* vector;
		/** length of vector  */
		index_t vlen;
	
	private:
		/** reference counter */
		int32_t* m_refcount;
};
}
#endif // __SGVECTOR_H__
