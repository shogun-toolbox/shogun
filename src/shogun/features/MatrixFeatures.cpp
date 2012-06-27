/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/features/MatrixFeatures.h>

namespace shogun {

template< class ST > CMatrixFeatures< ST >::CMatrixFeatures(int32_t size) : CFeatures(size)
{
	init();
}

/* TODO */
template< class ST > CFeatures* CMatrixFeatures< ST >::duplicate() const
{
	return NULL;
}

/* TODO */
template< class ST > CMatrixFeatures< ST >::~CMatrixFeatures()
{
}

/* TODO */
template< class ST > EFeatureType CMatrixFeatures< ST >::get_feature_type() const
{
	return F_UNKNOWN;
}

/* TODO */
template< class ST > EFeatureClass CMatrixFeatures< ST >::get_feature_class() const
{
	return C_UNKNOWN;
}

/* TODO */
template< class ST > int32_t CMatrixFeatures< ST >::get_size() const
{
	return 0;
}

/* TODO */
template< class ST > SGMatrix< ST > CMatrixFeatures< ST >::get_feature_matrix(
		int32_t num) const
{
	return SGMatrix< ST >();
}

/* TODO */
template< class ST > void CMatrixFeatures< ST >::set_feature_matrix(
		SGMatrix< ST > vector,
		int32_t num)
{
}

/* TODO */
template< class ST > SGNDArray< ST > CMatrixFeatures< ST >::get_feature_array() const
{
	return SGNDArray< ST >();
}

/* TODO */
template< class ST > void CMatrixFeatures< ST >::set_feature_array(SGNDArray< ST > array)
{
}

/* TODO */
template< class ST > void CMatrixFeatures< ST >::init()
{
}

template class CMatrixFeatures<bool>;
template class CMatrixFeatures<char>;
template class CMatrixFeatures<int8_t>;
template class CMatrixFeatures<uint8_t>;
template class CMatrixFeatures<int16_t>;
template class CMatrixFeatures<uint16_t>;
template class CMatrixFeatures<int32_t>;
template class CMatrixFeatures<uint32_t>;
template class CMatrixFeatures<int64_t>;
template class CMatrixFeatures<uint64_t>;
template class CMatrixFeatures<float32_t>;
template class CMatrixFeatures<float64_t>;
template class CMatrixFeatures<floatmax_t>;
} /* namespace shogun */
