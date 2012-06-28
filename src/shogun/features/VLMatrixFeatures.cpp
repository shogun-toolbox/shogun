/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/features/VLMatrixFeatures.h>

namespace shogun {

template< class ST > CVLMatrixFeatures< ST >::CVLMatrixFeatures(
		int32_t num_vec,
		int32_t num_feat)
: CFeatures(0)
{
	init();
}

/* TODO */
template< class ST > CFeatures* CVLMatrixFeatures< ST >::duplicate() const
{
	return NULL;
}

/* TODO */
template< class ST > CVLMatrixFeatures< ST >::~CVLMatrixFeatures()
{
}

/* TODO */
template< class ST > EFeatureType CVLMatrixFeatures< ST >::get_feature_type() const
{
	return F_UNKNOWN;
}

/* TODO */
template< class ST > EFeatureClass CVLMatrixFeatures< ST >::get_feature_class() const
{
	return C_UNKNOWN;
}

/* TODO */
template< class ST > int32_t CVLMatrixFeatures< ST >::get_size() const
{
	return 0;
}

/* TODO */
template< class ST > SGStringList< ST > CVLMatrixFeatures< ST >::get_feature_vector(
		int32_t num) const
{
	return SGStringList< ST >();
}

/* TODO */
template< class ST > void CVLMatrixFeatures< ST >::set_feature_vector(
		SGStringList< ST > const & vec,
		int32_t num)
{
}

/* TODO */
template< class ST > SGStringList< ST >* CVLMatrixFeatures< ST >::get_features() const
{
	return NULL;
}

/* TODO */
template< class ST > void CVLMatrixFeatures< ST >::set_features(
		SGStringList< ST >* features,
		int32_t num_vec)
{
}

/* TODO */
template< class ST > void CVLMatrixFeatures< ST >::init()
{
}

template class CVLMatrixFeatures<bool>;
template class CVLMatrixFeatures<char>;
template class CVLMatrixFeatures<int8_t>;
template class CVLMatrixFeatures<uint8_t>;
template class CVLMatrixFeatures<int16_t>;
template class CVLMatrixFeatures<uint16_t>;
template class CVLMatrixFeatures<int32_t>;
template class CVLMatrixFeatures<uint32_t>;
template class CVLMatrixFeatures<int64_t>;
template class CVLMatrixFeatures<uint64_t>;
template class CVLMatrixFeatures<float32_t>;
template class CVLMatrixFeatures<float64_t>;
template class CVLMatrixFeatures<floatmax_t>;

} /* namespace shogun */
