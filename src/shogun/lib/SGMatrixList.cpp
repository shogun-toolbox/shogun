/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Fernando Iglesias, Thoralf Klein, Shell Hu, 
 *          Koen van de Sande
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

template class SHOGUN_EXPORT SGMatrixList<bool>;
template class SHOGUN_EXPORT SGMatrixList<char>;
template class SHOGUN_EXPORT SGMatrixList<int8_t>;
template class SHOGUN_EXPORT SGMatrixList<uint8_t>;
template class SHOGUN_EXPORT SGMatrixList<int16_t>;
template class SHOGUN_EXPORT SGMatrixList<uint16_t>;
template class SHOGUN_EXPORT SGMatrixList<int32_t>;
template class SHOGUN_EXPORT SGMatrixList<uint32_t>;
template class SHOGUN_EXPORT SGMatrixList<int64_t>;
template class SHOGUN_EXPORT SGMatrixList<uint64_t>;
template class SHOGUN_EXPORT SGMatrixList<float32_t>;
template class SHOGUN_EXPORT SGMatrixList<float64_t>;
template class SHOGUN_EXPORT SGMatrixList<floatmax_t>;

} /* namespace shogun */
