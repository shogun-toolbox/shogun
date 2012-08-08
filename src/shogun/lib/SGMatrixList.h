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
		inline SGMatrix<T>& get_matrix(index_t index) const
		{
			return matrix_list[index];
		}

		/** operator overload to get a matrix for read & write access
		 *
		 * @param index matrix index, index must be less than
		 * num_matrices although no check is performed in the method
		 *
		 * @return the matrix at position index of the list
		 */
		inline SGMatrix<T>& operator[](index_t index) const
		{
			return matrix_list[index];
		}

	protected:
		/** copy data */
		virtual void copy_data(SGReferencedData const & orig);

		/** initialize empty data */
		virtual void init_data();

		/** free data */
		virtual void free_data();
	
	private:
		/** helper method of free_data */
		void cleanup_matrices();

	public:
		/** matrix list */
		SGMatrix<T>* matrix_list;

		/** number of matrices of matrix list */
		int32_t num_matrices;

}; /* class SGMatrixList */

} /* namespace shogun */

#endif /* define __SGMATRIX_LIST_H__ */
