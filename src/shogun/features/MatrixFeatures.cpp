/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Heiko Strathmann, Vladislav Horbatiuk,
 *          Bjoern Esser
 */

#include <shogun/features/MatrixFeatures.h>

namespace shogun {

template< class ST > MatrixFeatures< ST >::MatrixFeatures()
: Features(0)
{
	init();
}

template< class ST > MatrixFeatures< ST >::MatrixFeatures(
		int32_t num_vecs,
		int32_t num_feats)
: Features(0)
{
	init();
	m_features     = SGMatrixList< ST >(num_vecs);
	m_num_vectors  = num_vecs;
	m_num_features = num_feats;
}

template< class ST > MatrixFeatures< ST >::MatrixFeatures(
		SGMatrixList< ST > feats, int32_t num_feats)
: Features(0)
{
	init();
	set_features(feats, num_feats);
}

template< class ST > MatrixFeatures< ST >::MatrixFeatures(
		SGMatrix< ST > feats, int32_t feat_length, int32_t num_vecs)
: Features(0)
{
	REQUIRE(feats.num_cols == feat_length*num_vecs, "The number of columns of feats "
			"must be equal to feat_length times num_vecs\n");
	init();
	SGMatrixList< ST > feats_list = SGMatrixList< ST >::split(feats, num_vecs);
	set_features(feats_list, feats.num_rows);
}

/* TODO */
template< class ST > std::shared_ptr<Features> MatrixFeatures< ST >::duplicate() const
{
	return NULL;
}

template< class ST > MatrixFeatures< ST >::~MatrixFeatures()
{
	cleanup();
}

/* TODO */
template< class ST > EFeatureType MatrixFeatures< ST >::get_feature_type() const
{
	return F_UNKNOWN;
}

template< class ST > EFeatureClass MatrixFeatures< ST >::get_feature_class() const
{
	return C_MATRIX;
}

template< class ST > SGMatrix< ST > MatrixFeatures< ST >::get_feature_vector(
		int32_t num) const
{
	if ( num < 0 || num >= get_num_vectors() )
	{
		SG_ERROR("The index of the feature vector to get must be between "
			 "0 and %d (get_num_vectors()-1)\n", get_num_vectors()-1);
	}

	return m_features[num];
}

template< class ST > void MatrixFeatures< ST >::get_feature_vector_col(
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

template< class ST > void MatrixFeatures< ST >::set_feature_vector(
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

template< class ST > void MatrixFeatures< ST >::set_features(
		SGMatrixList< ST > features, int32_t num_feats)
{
	m_features     = features;
	m_num_vectors  = features.num_matrices;
	m_num_features = num_feats;
}

template< class ST > void MatrixFeatures< ST >::init()
{
	//SG_ADD(&m_num_vectors, "m_num_vectors", "Number of feature vectors");
	//SG_ADD(&m_num_features, "m_num_features",
	//		"Number of features per vector (optional)");
	//TODO add SG_ADD for SGMatrixList
	//SG_ADD(&m_features, "m_features", "Matrix features");

	m_num_vectors  = 0;
	m_num_features = 0;

	set_generic<ST>();
}

template< class ST > void MatrixFeatures< ST >::cleanup()
{
	m_features     = SGMatrixList< ST >();
	m_num_vectors  = 0;
	m_num_features = 0;
}

template< class ST > std::shared_ptr<MatrixFeatures< ST >> MatrixFeatures< ST >::obtain_from_generic(std::shared_ptr<Features> base_features)
{
	REQUIRE(base_features->get_feature_class() == C_MATRIX,
			"base_features must be of dynamic type CMatrixFeatures\n")

	return std::dynamic_pointer_cast<MatrixFeatures< ST >>(base_features);
}

template class MatrixFeatures<bool>;
template class MatrixFeatures<char>;
template class MatrixFeatures<int8_t>;
template class MatrixFeatures<uint8_t>;
template class MatrixFeatures<int16_t>;
template class MatrixFeatures<uint16_t>;
template class MatrixFeatures<int32_t>;
template class MatrixFeatures<uint32_t>;
template class MatrixFeatures<int64_t>;
template class MatrixFeatures<uint64_t>;
template class MatrixFeatures<float32_t>;
template class MatrixFeatures<float64_t>;
template class MatrixFeatures<floatmax_t>;

} /* namespace shogun */
