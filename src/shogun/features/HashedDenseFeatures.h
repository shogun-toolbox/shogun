/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#ifndef _HASHED_DENSEFEATURES_H__
#define _HASHED_DENSEFEATURES_H__

#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DotFeatures.h>

namespace shogun
{
template <class ST> class CDenseFeatures;
class CDotFeatures;

/** @brief This class is identical to the CDenseFeatures class
 * except that it hashes each dimension to a new feature space.
 */
template <class ST> class CHashedDenseFeatures : public CDotFeatures
{
public:

	/** constructor
	 *
	 * @param size cache size
	 * @param use_quadr whether to use quadratic features or not
	 * @param keep_lin_terms whether to maintain the linear terms in the computations
	 */
	CHashedDenseFeatures(int32_t size=0, bool use_quadr = false, bool keep_lin_terms = true);

	/** constructor
	 *
	 * @param feats	the dense features to use as a base
	 * @param d new feature space dimension
	 * @param use_quadr whether to use quadratic features or not
	 * @param keep_lin_terms whether to maintain the linear terms in the computations
	 */
	CHashedDenseFeatures(CDenseFeatures<ST>* feats, int32_t d, bool use_quadr = false,
			bool keep_lin_terms = true);

	/** constructor
	 *
	 * @param matrix feature matrix
	 * @param dm new feature space dimension
	 * @param use_quadr whether to use quadratic features or not
	 * @param keep_lin_terms whether to maintain the linear terms in the computations
	 */
	CHashedDenseFeatures(SGMatrix<ST> matrix, int32_t dm, bool use_quadr = false,
			bool keep_lin_terms = true);

	/** constructor
	 *
	 * @param src feature matrix
	 * @param num_feat number of features in matrix
	 * @param num_vec number of vectors in matrix
	 * @param d new feature space dimension
	 * @param use_quadr whether to use quadratic features or not
	 * @param keep_lin_terms whether to maintain the linear terms in the computations
	 */
	CHashedDenseFeatures(ST* src, int32_t num_feat, int32_t num_vec, int32_t d,
			bool use_quadr = false, bool keep_lin_terms = true);

	/** constructor loading features from file
	 *
	 * @param loader File object via which to load data
	 * @param d new feature space dimension
	 * @param use_quadr whether to use quadratic features or not
	 * @param keep_lin_terms whether to maintain the linear terms in the computations
	 */
	CHashedDenseFeatures(CFile* loader, int32_t d, bool use_quadr = false,
			bool keep_lin_terms = false);
	
	/** copy constructor */
	CHashedDenseFeatures(const CHashedDenseFeatures& orig);

	/** duplicate */
	virtual CFeatures* duplicate() const;

	/** destructor */
	virtual ~CHashedDenseFeatures();

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	virtual int32_t get_dim_feature_space() const;

	/** compute dot product between vector1 and vector2,
	 * appointed by their indices
	 *
	 * possible with subset
	 *
	 * @param vec_idx1 index of first vector
	 * @param df DotFeatures (of same kind) to compute dot product with
	 * @param vec_idx2 index of second vector
	 */
	virtual float64_t dot(int32_t vec_idx1, CDotFeatures* df,
			int32_t vec_idx2);

	/** compute dot product between vector1 and a dense vector
	 *
	 * possible with subset
	 *
	 * @param vec_idx1 index of first vector
	 * @param vec2 pointer to real valued vector
	 * @param vec2_len length of real valued vector
	 */
	virtual float64_t dense_dot(int32_t vec_idx1, const float64_t* vec2,
			int32_t vec2_len);

	/** add vector 1 multiplied with alpha to dense vector2
	 *
	 * possible with subset
	 *
	 * @param alpha scalar alpha
	 * @param vec_idx1 index of first vector
	 * @param vec2 pointer to real valued vector
	 * @param vec2_len length of real valued vector
	 * @param abs_val if true add the absolute value
	 */
	virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
			float64_t* vec2, int32_t vec2_len, bool abs_val = false);

	/** get number of non-zero features in vector
	 *
	 * @param num which vector
	 * @return number of non-zero features in vector
	 */
	virtual int32_t get_nnz_features_for_vector(int32_t num);

	/** iterate over the non-zero features
	 *
	 * call get_feature_iterator first, followed by get_next_feature and
	 * free_feature_iterator to cleanup
	 *
	 * possible with subset
	 *
	 * @param vector_index the index of the vector over whose components to
	 * 			iterate over
	 * @return feature iterator (to be passed to get_next_feature)
	 */
	virtual void* get_feature_iterator(int32_t vector_index);

	/** iterate over the non-zero features
	 *
	 * call this function with the iterator returned by get_first_feature
	 * and call free_feature_iterator to cleanup
	 *
	 * possible with subset
	 *
	 * @param index is returned by reference (-1 when not available)
	 * @param value is returned by reference
	 * @param iterator as returned by get_first_feature
	 * @return true if a new non-zero feature got returned
	 */
	virtual bool get_next_feature(int32_t& index, float64_t& value,
			void* iterator);

	/** clean up iterator
	 * call this function with the iterator returned by get_first_feature
	 *
	 * @param iterator as returned by get_first_feature
	 */
	virtual void free_feature_iterator(void* iterator);

	/** @return object name */
	virtual const char* get_name() const;

	/** get feature type
	 *
	 * @return templated feature type
	 */
	virtual EFeatureType get_feature_type() const;

	/** get feature class
	 *
	 * @return feature class DENSE
	 */
	virtual EFeatureClass get_feature_class() const;

	/** get number of feature vectors
	 *
	 * @return number of feature vectors
	 */
	virtual int32_t get_num_vectors() const;	

	/** get the hashed representation of the requested vector
	 *
	 * @param vec_idx the index of the vector
	 * @return the hashed representation of the vector
	 */
	SGSparseVector<ST> get_hashed_feature_vector(int32_t vec_idx);

	/** returns a hashed represenation of vector vec in a new feature space of dimension dim
	 *
	 * @param vec the vector to hash
	 * @param dim the size of the new dimension
	 * @param use_quadratic whether to use quadratic features or not
	 * @param keep_lin_terms whether to maintain the linear terms in the computations
	 * @return the hashed representation
	 */
	static SGSparseVector<ST> hash_vector(SGVector<ST> vec, int32_t dim, bool use_quadratic = false,
			bool keep_linear_terms = true);

protected:
	void init(CDenseFeatures<ST>* feats, int32_t d, bool use_quadr, bool keep_lin_terms);

protected:

	/** dense features */
	CDenseFeatures<ST>* dense_feats;

	/** new feature space dimension */
	int32_t dim;

	/** use quadratic feature or not */
	bool use_quadratic;

	/** keep linear terms or not */
	bool keep_linear_terms;
};
}

#endif // _HASHED_DENSEFEATURES_H__
