/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/lib/SGMatrixList.h>
#include <shogun/lib/memory.h>
#include <shogun/io/SGIO.h>

namespace shogun {

template <class T>
SGMatrixList<T>::SGMatrixList() : SGReferencedData()
{
	init_data();
}

template <class T>
SGMatrixList<T>::SGMatrixList(SGMatrix<T>* ml, int32_t nmats, bool ref_counting)
: SGReferencedData(ref_counting), matrix_list(ml), num_matrices(nmats)
{
}

template <class T>
SGMatrixList<T>::SGMatrixList(int32_t nmats, bool ref_counting)
: SGReferencedData(ref_counting), num_matrices(nmats)
{
	matrix_list = SG_MALLOC(SGMatrix<T>, nmats);
}

template <class T>
SGMatrixList<T>::SGMatrixList(SGMatrixList const & orig) : SGReferencedData(orig)
{
	copy_data(orig);
}

template <class T>
SGMatrixList<T>::~SGMatrixList()
{
	unref();
}

template <class T>
SGMatrix<T> SGMatrixList<T>::get_matrix(index_t index) const
{
	return matrix_list[index];
}

template <class T>
SGMatrix<T> SGMatrixList<T>::operator[](index_t index) const
{
	return matrix_list[index];
}

template <class T>
void SGMatrixList<T>::set_matrix(index_t index, const SGMatrix<T> matrix)
{
	matrix_list[index] = matrix;
}

template <class T>
void SGMatrixList<T>::copy_data(const SGReferencedData &orig)
{
	matrix_list  = ((SGMatrixList*) (&orig))->matrix_list;
	num_matrices = ((SGMatrixList*) (&orig))->num_matrices;
}

template <class T>
void SGMatrixList<T>::init_data()
{
	matrix_list  = NULL;
	num_matrices = 0;
}

template <class T>
void SGMatrixList<T>::free_data()
{
	SG_FREE(matrix_list);
	num_matrices = 0;
	matrix_list = NULL;
}

template <class T>
SGMatrixList<T> SGMatrixList<T>::split(SGMatrix<T> matrix, int32_t num_components)
{
	REQUIRE((matrix.num_cols % num_components) == 0,
		"The number of columns (%d) must be multiple of the number "
		"of components (%d).\n",
		matrix.num_cols, num_components);

	int32_t new_num_cols = matrix.num_cols / num_components;
	SGMatrixList<T> out(num_components);

	for ( int32_t i = 0 ; i < num_components ; ++i )
	{
		SGMatrix<T> new_matrix = SGMatrix<T>(matrix.num_rows, new_num_cols);

		for ( int32_t row = 0 ; row < matrix.num_rows ; ++row )
		{
			for ( int32_t col = 0 ; col < new_num_cols ; ++col )
				new_matrix(row, col) = matrix(row, int64_t(i)*new_num_cols + col);
		}

		out.set_matrix(i, new_matrix);
	}

	return out;
}

template class SGMatrixList<bool>;
template class SGMatrixList<char>;
template class SGMatrixList<int8_t>;
template class SGMatrixList<uint8_t>;
template class SGMatrixList<int16_t>;
template class SGMatrixList<uint16_t>;
template class SGMatrixList<int32_t>;
template class SGMatrixList<uint32_t>;
template class SGMatrixList<int64_t>;
template class SGMatrixList<uint64_t>;
template class SGMatrixList<float32_t>;
template class SGMatrixList<float64_t>;
template class SGMatrixList<floatmax_t>;

} /* namespace shogun */
