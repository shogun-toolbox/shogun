/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef __SGMATRIX_LIST_H__
#define __SGMATRIX_LIST_H__

#include <shogun/lib/config.h>

#include <shogun/lib/SGReferencedData.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

/** @brief shogun matrix list */
template<class T> class SGMatrixList : public SGReferencedData
{
	public:
		/** default constructor */
		SGMatrixList();

		/** constructor for setting parameters */
		SGMatrixList(SGMatrix<T>* ml, int32_t nmats, bool ref_counting = true);

		/** constructor to create a new matrix list in memory */
		SGMatrixList(int32_t nmats, bool ref_counting = true);

		/** copy constructor */
		SGMatrixList(SGMatrixList const & orig);

		/** destructor */
		virtual ~SGMatrixList();

		/** get a matrix of the list
		 *
		 * @param index matrix index, index must be less than
		 * num_matrices although no check is performed in the method
		 *
		 * @return the matrix at position index of the list
		 */
		SGMatrix<T> get_matrix(index_t index) const;

		/** operator overload to get a matrix for read & write access
		 *
		 * @param index matrix index, index must be less than
		 * num_matrices although no check is performed in the method
		 *
		 * @return the matrix at position index of the list
		 */
		SGMatrix<T> operator[](index_t index) const;

		/** set a matrix of the list
		 *
		 * @param index matrix index, index must be less than
		 * num_matrices although no check is performed in the method
		 * @param matrix matrix to set at index
		 */
		void set_matrix(index_t index, const SGMatrix<T> matrix);

		/**
		 * divide the matrix into a list of matrices. Each of the new
		 * matrices has the same number of rows as the original so the
		 * splits to the original matrix are done column-wise.
		 *
		 * @param matrix matrix to split
		 * @param num_components number of new matrices
		 *
		 * @return list of matrices
		 */
		static SGMatrixList<T> split(SGMatrix<T> matrix, int32_t num_components);

	protected:
		/** copy data */
		virtual void copy_data(const SGReferencedData &orig);

		/** initialize empty data */
		virtual void init_data();

		/** free data */
		virtual void free_data();

	public:
		/** matrix list */
		SGMatrix<T>* matrix_list;

		/** number of matrices of matrix list */
		int32_t num_matrices;

}; /* class SGMatrixList */

} /* namespace shogun */

#endif /* define __SGMATRIX_LIST_H__ */
