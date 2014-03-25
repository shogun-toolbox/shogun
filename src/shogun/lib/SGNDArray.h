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
#include <shogun/io/SGIO.h>

namespace shogun
{
/** @brief shogun n-dimensional array */
template<class T> class SGNDArray : public SGReferencedData
{
	public:
		/** default constructor */
		SGNDArray();

		/** constructor for setting params */
		SGNDArray(T* a, index_t* d, index_t nd, bool ref_counting=true);

		/** constructor to create new ndarray in memory */
		SGNDArray(index_t* d, index_t nd, bool ref_counting=true);

		/** copy constructor */
		SGNDArray(const SGNDArray &orig);

		/** empty destructor */
		virtual ~SGNDArray();

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
};
}
#endif // __SGNDARRAY_H__
