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

template< class ST > CMatrixFeatures< ST >::CMatrixFeatures()
: CFeatures(0)
{
	init();
}

template< class ST > CMatrixFeatures< ST >::CMatrixFeatures(
		int32_t num_vec,
		int32_t num_feat)
: CFeatures(0), m_num_vectors(num_vec), m_num_features(num_feat)
{
	init();

	m_features = SG_MALLOC(SGMatrix< ST >, num_vec);
}

template< class ST > CMatrixFeatures< ST >::CMatrixFeatures(
		SGMatrix< ST >* feats,
		int32_t num)
: CFeatures(0)
{
	init();
	set_features(feats, num);
}

/* TODO */
template< class ST > CFeatures* CMatrixFeatures< ST >::duplicate() const
{
	return NULL;
}

template< class ST > CMatrixFeatures< ST >::~CMatrixFeatures()
{
	cleanup();
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

template< class ST > SGMatrix< ST > CMatrixFeatures< ST >::get_feature_vector(
		int32_t num) const
{
	if ( num < 0 || num >= get_num_vectors() )
	{
		SG_ERROR("The index of the feature vector to get must be between "
			 "0 and %d (get_num_vectors()-1)\n", get_num_vectors()-1);
	}

	return m_features[num];
}

template< class ST > void CMatrixFeatures< ST >::get_feature_vector_col(
		SGVector< ST > out, 
		int32_t num, 
		int32_t col) const
{
	if ( num < 0 || num >= get_num_vectors() )
	{
		SG_ERROR("The index of the feature vector to get must be between "
			 "0 and %d (get_num_vectors()-1)\n", get_num_vectors()-1);
	}

	// Shorthands for the dimensions of the feature vector to get
	int32_t num_cols = m_features[num].num_cols;
	int32_t num_rows = m_features[num].num_rows;

	if ( col < 0 || col >= num_cols )
	{
		SG_ERROR("The index of the column to get must be between "
			 "0 and %d (#columns of the feature vector)\n", num_cols);
	}

	if ( out.vlen < get_num_features() )
	{
		SG_ERROR("The vector out must have space to hold at least "
			 "%d (get_num_features()) elements\n", get_num_features());
	}

	int32_t start = col*num_rows;
	for ( int32_t i = 0 ; i < get_num_features(); ++i )
	{
		out[i] = m_features[num][start + i];
	}
}

template< class ST > void CMatrixFeatures< ST >::set_feature_vector(
		SGMatrix< ST > const & vec,
		int32_t num)
{
	if ( num < 0 || num >= get_num_vectors() )
	{
		SG_ERROR("The index of the feature vector to set must be between "
			 "0 and %d (get_num_vectors()-1)\n", get_num_vectors()-1);
	}

	if ( vec.num_rows != get_num_features() )
	{
		SG_ERROR("The feature vector to set must have the same features "
			 "as the rest of the MatrixFeatures, %d "
			 "(get_num_features())\n", get_num_features());
	}

	m_features[num] = vec;
}

template< class ST > SGMatrix< ST >* CMatrixFeatures< ST >::get_features(
		int32_t& num_vec) const
{
	num_vec = get_num_vectors();
	return m_features;
}

template< class ST > void CMatrixFeatures< ST >::set_features(
		SGMatrix< ST >* features,
		int32_t num_vec)
{
	cleanup();

	m_features = features;

	m_num_vectors  = num_vec;
	if ( num_vec > 0 )
	{
		m_num_features = features[0].num_rows;
	}
	else
	{
		SG_ERROR("Cannot set empty features without at least one "
			  "feature vector\n");
	}
}

template< class ST > void CMatrixFeatures< ST >::init()
{
	SG_ADD(&m_num_vectors, "m_num_vectors", "Number of feature vectors", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_features, "m_num_features", "Number of features of each vector", MS_NOT_AVAILABLE);
	//TODO add m_features for serialization
	
	m_features = NULL;
}

template< class ST > void CMatrixFeatures< ST >::cleanup()
{
	cleanup_feature_vectors(0, get_num_vectors()-1);

	m_num_vectors  = 0;
	m_num_features = 0;
	SG_FREE(m_features);
	m_features = NULL;
}

template< class ST > void CMatrixFeatures< ST >::cleanup_feature_vectors(
		int32_t start, 
		int32_t stop)
{
	if ( m_features && get_num_vectors() )
	{
		ASSERT(0 <= start && start < get_num_vectors());
		ASSERT(stop < get_num_vectors());
		ASSERT(stop >= start);

		for ( int32_t i = start ; i <= stop ; ++i )
		{
			// Explicit call to the destructor in case
			// an in-place constructor was used
			m_features[i].~SGMatrix();
		}
	}
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
