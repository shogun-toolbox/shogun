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

#include <shogun/io/SGIO.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGReferencedData.h>

namespace shogun
{
/** @brief shogun vector */
template<class T> class SGVector : public SGReferencedData
{
	public:
		/** default constructor */
		SGVector();

		/** constructor for setting params */
		SGVector(T* v, index_t len, bool ref_counting=true);

		/** constructor to create new vector in memory */
		SGVector(index_t len, bool ref_counting=true);

		/** copy constructor */
		SGVector(const SGVector &orig);

		/** empty destructor */
		virtual ~SGVector();

		/** fill vector with zeros */
		void zero();

		/** set vector to a constant
		 *
		 * @param const_elem - value to set vector to
		 */
		void set_const(T const_elem);

		/** range fill a vector with start...start+len-1
		 * 
		 * @param start - value to be assigned to first element of vector
		 */
		void range_fill(T start=0);

		/** create random vector
		 *
		 * @param min_value [min_value,max_value]
		 * @param max_value
		 */
		void random(T min_value, T max_value);

		/** random permutate */
		void randperm();

		/** clone vector */
		SGVector<T> clone() const;

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
		const T& get_element(index_t index);

		/** set vector element at index 'index' return false in case of trouble
		 *
		 * @param p_element vector element to set
		 * @param index index
		 * @return if setting was successful
		 */
		void set_element(const T& p_element, index_t index);

		/** resize vector
		 *
		 * @param n new size
		 * @return if resizing was successful
		 */
		void resize_vector(int32_t n);

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

		/** add vector to current vector
		 *
		 * @param x add vector x to current vector
		 */
		void add(const SGVector<T> x);

		SGVector<T> operator+ (SGVector<T> x)
		{
			ASSERT(x.vector && vector);
			ASSERT(x.vlen == vlen);

			SGVector<T> result=clone();
			result.add(x);
			return result;
		}

		SGVector<T> operator+= (SGVector<T> x)
		{
			add(x);
			return *this;
		}

		/** display array size */
		void display_size() const;

		/** display array */
		void display_vector() const;

	protected:
		/** needs to be overridden to copy data */
		virtual void copy_data(const SGReferencedData &orig);

		/** needs to be overridden to initialize empty data */
		virtual void init_data();

		/** needs to be overridden to free data */
		virtual void free_data();

	public:
		/** vector  */
		T* vector;
		/** length of vector  */
		index_t vlen;
};
}
#endif // __SGVECTOR_H__
