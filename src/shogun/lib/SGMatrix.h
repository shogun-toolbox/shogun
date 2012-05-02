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
#ifndef __SGMATRIX_H__
#define __SGMATRIX_H__

#include <shogun/lib/config.h>
#include <shogun/lib/DataType.h>

namespace shogun
{
/** @brief shogun matrix */
template<class T> class SGMatrix
{
	public:
		/** default constructor */
		SGMatrix() : matrix(NULL), num_rows(0), num_cols(0), do_free(false) { }

		/** constructor for setting params */
		SGMatrix(T* m, index_t nrows, index_t ncols, bool free_mat=false)
			: matrix(m), num_rows(nrows), num_cols(ncols), do_free(free_mat) { }

		/** constructor to create new matrix in memory */
		SGMatrix(index_t nrows, index_t ncols, bool free_mat=false)
			: num_rows(nrows), num_cols(ncols), do_free(free_mat)
		{
			matrix=SG_MALLOC(T, nrows*ncols);
		}

		/** copy constructor */
		SGMatrix(const SGMatrix &orig)
			: matrix(orig.matrix), num_rows(orig.num_rows),
			num_cols(orig.num_cols), do_free(orig.do_free) { }

		/** empty destructor */
		virtual ~SGMatrix()
		{
		}

		/** free matrix */
		virtual void free_matrix()
		{
			if (do_free)
				SG_FREE(matrix);

			matrix=NULL;
			do_free=false;
			num_rows=0;
			num_cols=0;
		}

		/** destroy matrix */
		virtual void destroy_matrix()
		{
			do_free=true;
			free_matrix();
		}

		/** get a column vector
		 * @param col column index
		 */
		T* get_column_vector(index_t col) const
		{
			return &matrix[col*num_rows];
		}

        /** operator overload for matrix read only access
         * @param i_row
         * @param i_col
         */
        inline const T& operator()(index_t i_row, index_t i_col) const
        {
            return matrix[i_col*num_rows + i_row];
        }

		/** operator overload for matrix read only access
		 * @param index to access
		 */
		inline const T& operator[](index_t index) const
		{
			return matrix[index];
		}

        /** operator overload for matrix r/w access
         * @param i_row
         * @param i_col
         */
        inline T& operator()(index_t i_row, index_t i_col)
        {
            return matrix[i_col*num_rows + i_row];
        }

		/** operator overload for matrix r/w access
		 * @param index to access
		 */
		inline T& operator[](index_t index)
		{
			return matrix[index];
		}

		/** set matrix to a constant */
		void set_const(T const_elem)
		{
			for (index_t i=0; i<num_rows*num_cols; i++)
				matrix[i]=const_elem ;
		}

		/** fill matrix with zeros */
		void zero()
		{
			if (matrix && (num_rows*num_cols))
				set_const(0);
		}

	public:
		/** matrix  */
		T* matrix;
		/** number of rows of matrix  */
		index_t num_rows;
		/** number of columns of matrix  */
		index_t num_cols;
		/** whether matrix needs to be freed */
		bool do_free;
};
}
#endif // __SGMATRIX_H__
