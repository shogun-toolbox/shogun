/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#ifndef _HASHEDDOCDOTFEATURES__H__
#define _HASHEDDOCDOTFEATURES__H__

#include <shogun/features/DotFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/converter/HashedDocConverter.h>
#include <shogun/lib/Tokenizer.h>

namespace shogun {
template<class ST> class CStringFeatures;
template<class ST> class SGMatrix;
class CDotFeatures;
class CHashedDocConverter;
class CTokenizer;

class CHashedDocDotFeatures: public CDotFeatures
{
public:

	/** constructor
	 *
	 * @param hash_bits the number of bits of the hash. Means a dimension of size 2^(hash_bits).
	 * @param docs the document collection
	 * @param tzer the tokenizer to use on the documents
	 * @param size cache size
	 */
	CHashedDocDotFeatures(int32_t hash_bits=0, CStringFeatures<char>* docs=NULL, 
			CTokenizer* tzer=NULL, int32_t size=0);

	/** copy constructor */
	CHashedDocDotFeatures(const CHashedDocDotFeatures& orig);

	/** constructor
	 *
	 * @param loader File object via which to load data
	 */
	CHashedDocDotFeatures(CFile* loader);

	/** destructor */
	virtual ~CHashedDocDotFeatures();
	
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
	 * @param vec_idx1 index of first vector
	 * @param df DotFeatures (of same kind) to compute dot product with
	 * @param vec_idx2 index of second vector
	 */
	virtual float64_t dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2);

	/** compute dot product between vector1 and a dense vector
	 *
	 * @param vec_idx1 index of first vector
	 * @param vec2 dense vector
	 */
	virtual float64_t dense_dot_sgvec(int32_t vec_idx1, const SGVector<float64_t> vec2);

	/** compute dot product between vector1 and a dense vector
	 *
	 * @param vec_idx1 index of first vector
	 * @param vec2 pointer to real valued vector
	 * @param vec2_len length of real valued vector
	 */
	virtual float64_t dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len);

	/** add vector 1 multiplied with alpha to dense vector2
	 *
	 * @param alpha scalar alpha
	 * @param vec_idx1 index of first vector
	 * @param vec2 pointer to real valued vector
	 * @param vec2_len length of real valued vector
	 * @param abs_val if true add the absolute value
	 */
	virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val=false);

	/** get number of non-zero features in vector
	 *
	 * (in case accurate estimates are too expensive overestimating is OK)
	 *
	 * @param num which vector
	 * @return number of sparse features in vector
	 */
	virtual int32_t get_nnz_features_for_vector(int32_t num);

	/** iterate over the non-zero features
	 *
	 * call get_feature_iterator first, followed by get_next_feature and
	 * free_feature_iterator to cleanup
	 * NOT IMPLEMENTED
	 *
	 * @param vector_index the index of the vector over whose components to
	 * 			iterate over
	 * @return feature iterator (to be passed to get_next_feature)
	 */
	virtual void* get_feature_iterator(int32_t vector_index);

	/** iterate over the non-zero features
	 * NOT IMPLEMENTED
	 *
	 * call this function with the iterator returned by get_feature_iterator
	 * and call free_feature_iterator to cleanup
	 *
	 * @param index is returned by reference (-1 when not available)
	 * @param value is returned by reference
	 * @param iterator as returned by get_feature_iterator
	 * @return true if a new non-zero feature got returned
	 */
	virtual bool get_next_feature(int32_t& index, float64_t& value, void* iterator);

	/** clean up iterator
	 * call this function with the iterator returned by get_feature_iterator
	 * NOT IMPLEMENTED
	 *
	 * @param iterator as returned by get_feature_iterator
	 */
	virtual void free_feature_iterator(void* iterator);

	/** set the document collection to work on
	 *
	 * @param docs the document collection
	 */
	void set_doc_collection(CStringFeatures<char>* docs);

	virtual const char* get_name() const;

	/** duplicate feature object
	 *
	 * @return feature object
	 */
	virtual CFeatures* duplicate() const;

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

	/** Helper method to calculate the murmur hash of a
	 * token and restrict it to a specified dimension range.
	 *
	 * @param token pointer to the token
	 * @param length the length of the token
	 * @param num_bits the number of bits to maintain in the hash
	 * @param seed a seed for the hash
	 */
	static uint32_t calculate_token_hash(char* token, int32_t length, 
			int32_t num_bits, uint32_t seed);

private:
	void init(int32_t hash_bits, CStringFeatures<char>* docs, CTokenizer* tzer);

protected:
	/** the document collection*/
	CStringFeatures<char>* doc_collection;

	/** number of bits of hash */
	int32_t num_bits;

	/** tokenizer */
	CTokenizer* tokenizer;
};
}

#endif
