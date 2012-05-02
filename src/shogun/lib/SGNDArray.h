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
#include <shogun/lib/DataType.h>

namespace shogun
{
/** @brief shogun n-dimensional array */
template<class T> class SGNDArray
{
	public:
		/** default constructor */
		SGNDArray() : array(NULL), dims(NULL), num_dims(0), do_free(false) { }

		/** constructor for setting params */
		SGNDArray(T* a, index_t* d, index_t nd, bool do_free_ndarray = false)
		    : array(a), dims(d), num_dims(nd), do_free(do_free_ndarray) { }

		/** constructor to create new ndarray in memory */
		SGNDArray(index_t* d, index_t nd, bool do_free_ndarray = false)
			: dims(d), num_dims(nd), do_free(do_free_ndarray)
		{
			index_t tot = 1;
			for (int32_t i=0; i<nd; i++)
				tot *= dims[i];
			array=SG_MALLOC(T, tot);
		}

		/** copy constructor */
		SGNDArray(const SGNDArray &orig)
		    : array(orig.array), dims(orig.dims), num_dims(orig.num_dims),
		    do_free(orig.do_free) { }

		/** empty destructor */
		virtual ~SGNDArray()
		{
		}

		/** free ndarray */
		virtual void free_ndarray()
		{
			if (do_free)
				SG_FREE(array);

			SG_FREE(dims);

			array     = NULL;
			dims      = NULL;
			num_dims  = 0;
		}


		/** destroy ndarray */
		virtual void destroy_ndarray()
		{
			do_free = true;
			free_ndarray();
		}

		/** get a matrix formed by the two first dimensions
		 *
		 * @param  matIdx matrix index
		 * @return pointer to the matrix
		 */
		T* get_matrix(index_t matIdx) const
		{
			ASSERT(array && dims && num_dims > 2 && dims[2] > matIdx);
			return &array[matIdx*dims[0]*dims[1]];
		}

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

		/** transposes a matrix formed by the two first dimensions
		 *
		 * @param matIdx matrix index
		 */
		void transpose_matrix(index_t matIdx) const
		{
			ASSERT(array && dims && num_dims > 2 && dims[2] > matIdx);

			T aux;
			// Index to acces directly the elements of the matrix of interest
			int32_t idx = matIdx*dims[0]*dims[1];

			for (int32_t i=0; i<dims[0]; i++)
				for (int32_t j=0; j<i-1; j++)
				{
					aux = array[idx + i + j*dims[0]];
					array[idx + i + j*dims[0]] = array[idx + j + i*dims[0]];
					array[idx + j + i*dims[1]] = aux;
				}

			// Swap the sizes of the two first dimensions
			index_t auxDim = dims[0];
			dims[0] = dims[1];
			dims[1] = auxDim;
		}

	public:
		/** array  */
		T* array;
		/** dimension sizes */
		index_t* dims;
		/** number of dimensions  */
		index_t num_dims;
		/** whether ndarry needs to be freed */
		bool do_free;
};
}
#endif // __SGNDARRAY_H__
