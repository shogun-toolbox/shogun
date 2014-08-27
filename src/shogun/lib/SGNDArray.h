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

#ifndef __SGNDARRAY_H__
#define __SGNDARRAY_H__

#include <shogun/lib/config.h>

#include <shogun/lib/SGReferencedData.h>
#include <shogun/lib/SGVector.h>
#include <shogun/io/SGIO.h>

namespace shogun
{
/** @brief shogun n-dimensional array */
template<class T> class SGNDArray : public SGReferencedData
{
	public:
		/** default constructor */
		SGNDArray();

		/** constructor for setting params 
		 *
		 * @param a data of the array
		 * @param d dimentions of the array
		 * @param nd number of dimentions
		 * @param ref_counting if true, count the reference
		 */
		SGNDArray(T* a, index_t* d, index_t nd, bool ref_counting=true);

		/** constructor to create new ndarray in memory
		 * 
		 * @param d dimentions of the array
		 * @param nd number of dimentions
		 * @param ref_counting if true, count the reference
		 */
		SGNDArray(index_t* d, index_t nd, bool ref_counting=true);
		
		/** constructor to create new ndarray of given bases
		 * 
		 * @param d dimentions of the array
		 * @param ref_counting if true, count the reference
		 */
		SGNDArray(SGVector<index_t> d, bool ref_counting=true);

		/** copy constructor 
		 * 
		 * @param orig the original N-d array
		 */
		SGNDArray(const SGNDArray &orig);

		/** empty destructor */
		virtual ~SGNDArray();
		
		/** @return the cloned N-d array */
		SGNDArray<T> clone() const;

		/** get a matrix formed by the two first dimensions
		 *
		 * @param  matIdx matrix index
		 * @return pointer to the matrix
		 */
		T* get_matrix(index_t matIdx) const
		{
			ASSERT(array && dims && num_dims > 2 && dims[2] > matIdx)
			return &array[int64_t(matIdx)*int64_t(dims[0])*dims[1]];
		}

		/** transposes a matrix formed by the two first dimensions
		 *
		 * @param matIdx matrix index
		 */
		void transpose_matrix(index_t matIdx) const;
	
		/** operator overload for ndarray read only access
		 *
		 * @param index to access
		 */
		inline const T& operator[](index_t index) const
		{
			return array[index];
		}

		/** operator overload for ndarray r/w access
		 *
		 * @param index to access
		 */
		inline T& operator[](index_t index)
		{
			return array[index];
		}
	
		/** @return dimentions of the N-d array*/
		SGVector<index_t> get_dimensions() const;

		/** set N-d array to a constant
		 *
		 * @param const_elem constant value to set N-d array to 
		 */
		void set_const(T const_elem);
        
		/** operator overload of multiplication assignment
		 *
		 * @param val a scalar value to multiply
		 * @return the result N-d array
		 */
		SGNDArray<T>& operator*=(T val);
		
		/** operator overload of addition assignment
		 *
		 * @param ndarray N-d array to add
		 * @returns the result N-d array
		 */
		SGNDArray<T>& operator+=(SGNDArray& ndarray);
		
		/** operator overload of subtruction assignment
		 *
		 * @param ndarray N-d array to add
		 * @returns the result N-d array
		 */
		SGNDArray<T>& operator-=(SGNDArray& ndarray);
		
		/** find the maximum value of the elements
		 *
		 * @param max_at the index of the maximum element, index is in 1-d flattend array.
		 * If there are multiple maximum element, return the last index.
		 * @return the maximum value
		 */
		T max_element(index_t& max_at);
		
		/** expand to a big size array
		 *
		 * @param big_array the target big size array
		 * @param axes the axis where the current ndarray will be replicated
		 */
		void expand(SGNDArray& big_array, SGVector<index_t>& axes);
		
		/** get the value at index
		 *
		 * @param the index of the N-d array
		 * @return the value at index
		 */
		T get_value(SGVector<index_t> index) const;
		
		/** get the next index from the current one
		 *
		 * @param curr_index the current index
		 */
		void next_index(SGVector<index_t>& curr_index) const;

	protected:

		/** copy data */
		virtual void copy_data(const SGReferencedData &orig);

		/** init data */
		virtual void init_data();

		/** free data */
		virtual void free_data();

	public:

		/** array  */
		T* array;

		/** dimension sizes */
		index_t* dims;

		/** number of dimensions  */
		index_t num_dims;

		/** the flatten length of the N-d array */
		index_t len_array;
};
}
#endif // __SGNDARRAY_H__
