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
#include <shogun/lib/SGReferencedData.h>

namespace shogun
{
/** @brief shogun matrix */
template<class T> class SGMatrix : public SGReferencedData
{
	public:
		/** default constructor */
		SGMatrix() : SGReferencedData(false)
		{
			init_data();
		}

		/** constructor for setting params */
		SGMatrix(T* m, index_t nrows, index_t ncols, bool ref_counting=true)
			: SGReferencedData(ref_counting), matrix(m),
			num_rows(nrows), num_cols(ncols) { }

		/** constructor to create new matrix in memory */
		SGMatrix(index_t nrows, index_t ncols, bool ref_counting=true)
			: SGReferencedData(ref_counting), num_rows(nrows), num_cols(ncols)
		{
			matrix=SG_MALLOC(T, ((int64_t) nrows)*ncols);
		}

		/** copy constructor */
		SGMatrix(const SGMatrix &orig) : SGReferencedData(orig)
		{
			copy_data(orig);
		}

		/** empty destructor */
		virtual ~SGMatrix()
		{
			unref();
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

		/** clone matrix */
		SGMatrix<T> clone()
		{
			return SGMatrix<T>(clone_matrix(matrix, num_rows, num_cols),
					num_rows, num_cols);
		}

		/** clone vector */
		template <class VT>
		static VT* clone_matrix(const VT* matrix, int32_t nrows, int32_t ncols)
		{
			VT* result = SG_MALLOC(VT, int64_t(nrows)*ncols);
			for (int64_t i=0; i<int64_t(nrows)*ncols; i++)
				result[i]=matrix[i];

			return result;
		}

	protected:
		/** needs to be overridden to copy data */
		virtual void copy_data(const SGReferencedData &orig)
		{
			matrix=((SGMatrix*)(&orig))->matrix;
			num_rows=((SGMatrix*)(&orig))->num_rows;
			num_cols=((SGMatrix*)(&orig))->num_cols;
		}

		/** needs to be overridden to initialize empty data */
		virtual void init_data()
		{
			matrix=NULL;
			num_rows=0;
			num_cols=0;
		}

		/** needs to be overridden to free data */
		virtual void free_data()
		{
			SG_FREE(matrix);
			matrix=NULL;
			num_rows=0;
			num_cols=0;
		}

	public:
		/** matrix  */
		T* matrix;
		/** number of rows of matrix  */
		index_t num_rows;
		/** number of columns of matrix  */
		index_t num_cols;
};
}
#endif // __SGMATRIX_H__
