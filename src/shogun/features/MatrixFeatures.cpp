/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando Jose Iglesias Garcia
 * Copyright (C) 2012 Fernando Jose Iglesias Garcia
 */

#include <features/MatrixFeatures.h>

namespace shogun {

template< class ST > CMatrixFeatures< ST >::CMatrixFeatures()
: CFeatures(0)
{
	init();
}

template< class ST > CMatrixFeatures< ST >::CMatrixFeatures(
		int32_t num_vecs,
		int32_t num_feats)
: CFeatures(0)
{
	init();
	m_features     = SGMatrixList< ST >(num_vecs);
	m_num_vectors  = num_vecs;
	m_num_features = num_feats;
}

template< class ST > CMatrixFeatures< ST >::CMatrixFeatures(
		SGMatrixList< ST > feats, int32_t num_feats)
: CFeatures(0)
{
	init();
	set_features(feats, num_feats);
}

template< class ST > CMatrixFeatures< ST >::CMatrixFeatures(
		SGMatrix< ST > feats, int32_t feat_length, int32_t num_vecs)
: CFeatures(0)
{
	REQUIRE(feats.num_cols == feat_length*num_vecs, "The number of columns of feats "
			"must be equal to feat_length times num_vecs\n");
	init();
	SGMatrixList< ST > feats_list = SGMatrixList< ST >::split(feats, num_vecs);
	set_features(feats_list, feats.num_rows);
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

template< class ST > EFeatureClass CMatrixFeatures< ST >::get_feature_class() const
{
	return C_MATRIX;
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
		SGMatrix< ST > const vec,
		int32_t num)
{
	if ( num < 0 || num >= get_num_vectors() )
	{
		SG_ERROR("The index of the feature vector to set must be between "
			 "0 and %d (get_num_vectors()-1)\n", get_num_vectors()-1);
	}

	if ( get_num_features() != 0 && vec.num_rows != get_num_features() )
	{
		SG_ERROR("The feature vector to set must have the same features "
			 "as the rest of the MatrixFeatures, %d "
			 "(get_num_features())\n", get_num_features());
	}

	m_features.set_matrix(num, vec);
}

template< class ST > void CMatrixFeatures< ST >::set_features(
		SGMatrixList< ST > features, int32_t num_feats)
{
	m_features     = features;
	m_num_vectors  = features.num_matrices;
	m_num_features = num_feats;
}

template< class ST > void CMatrixFeatures< ST >::init()
{
	SG_ADD(&m_num_vectors, "m_num_vectors", "Number of feature vectors",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_num_features, "m_num_features",
			"Number of features per vector (optional)", MS_NOT_AVAILABLE);
	//TODO add SG_ADD for SGMatrixList
	//SG_ADD(&m_features, "m_features", "Matrix features", MS_NOT_AVAILABLE);

	m_num_vectors  = 0;
	m_num_features = 0;

	set_generic<ST>();
}

template< class ST > void CMatrixFeatures< ST >::cleanup()
{
	m_features     = SGMatrixList< ST >();
	m_num_vectors  = 0;
	m_num_features = 0;
}

template< class ST > CMatrixFeatures< ST >* CMatrixFeatures< ST >::obtain_from_generic(CFeatures* const base_features)
{
	REQUIRE(base_features->get_feature_class() == C_MATRIX,
			"base_features must be of dynamic type CMatrixFeatures\n")

	return (CMatrixFeatures< ST >*) base_features;
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
