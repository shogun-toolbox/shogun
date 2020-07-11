/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#ifndef _HASHEDDOCDOTFEATURES__H__
#define _HASHEDDOCDOTFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/features/DotFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/converter/HashedDocConverter.h>
#include <shogun/lib/Tokenizer.h>

namespace shogun {
template<class ST> class StringFeatures;
template<class ST> class SGMatrix;
class DotFeatures;
class HashedDocConverter;
class Tokenizer;

/** @brief This class can be used to provide on-the-fly vectorization of a document collection.
 * Like in the standard Bag-of-Words representation, this class considers each document as a collection of tokens,
 * which are then hashed into a new feature space of a specified dimension.
 * This class is very flexible and allows the user to specify the tokenizer used to tokenize each document,
 * specify whether the results should be normalized with regards to the sqrt of the document size, as well
 * as to specify whether he wants to combine different tokens.
 * The latter implements a k-skip n-grams approach, meaning that you can combine up to n tokens, while skipping up to k.
 * Eg. for the tokens ["a", "b", "c", "d"], with n_grams = 2 and skips = 2, one would get the following combinations :
 * ["a", "ab", "ac" (skipped 1), "ad" (skipped 2), "b", "bc", "bd" (skipped 1), "c", "cd", "d"].
 */
class HashedDocDotFeatures: public DotFeatures
{
public:

	HashedDocDotFeatures();

	/** constructor
	 *
	 * @param hash_bits the number of bits of the hash. Means a dimension of size 2^(hash_bits).
	 * @param docs the document collection
	 * @param tzer the tokenizer to use on the documents
	 */
	HashedDocDotFeatures(int32_t hash_bits, const std::shared_ptr<StringFeatures<char>>& docs,
			const std::shared_ptr<Tokenizer>& tzer);

	/** constructor
	 *
	 * @param hash_bits the number of bits of the hash. Means a dimension of size 2^(hash_bits).
	 * @param docs the document collection
	 * @param tzer the tokenizer to use on the documents
	 * @param normalize whether or not to normalize the result of the dot products
	 */
	HashedDocDotFeatures(int32_t hash_bits, const std::shared_ptr<StringFeatures<char>>& docs,
			const std::shared_ptr<Tokenizer>& tzer, bool normalize);

	/** constructor
	 *
	 * @param hash_bits the number of bits of the hash. Means a dimension of size 2^(hash_bits).
	 * @param docs the document collection
	 * @param tzer the tokenizer to use on the documents
	 * @param normalize whether or not to normalize the result of the dot products
	 * @param n_grams max number of consecutive tokens to hash together (extra features)
	 * @param skips max number of tokens to skip when combining tokens
	 */
	HashedDocDotFeatures(int32_t hash_bits, const std::shared_ptr<StringFeatures<char>>& docs,
			const std::shared_ptr<Tokenizer>& tzer, bool normalize, int32_t n_grams, int32_t skips);

	/** copy constructor */
	HashedDocDotFeatures(const HashedDocDotFeatures& orig);

	/** constructor
	 *
	 * @param loader File object via which to load data
	 */
	HashedDocDotFeatures(const std::shared_ptr<File>& loader);

	/** destructor */
	~HashedDocDotFeatures() override;

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	int32_t get_dim_feature_space() const override;

	/** compute dot product between vector1 and vector2,
	 * appointed by their indices
	 *
	 * @param vec_idx1 index of first vector
	 * @param df DotFeatures (of same kind) to compute dot product with
	 * @param vec_idx2 index of second vector
	 */
	float64_t dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df, int32_t vec_idx2) const override;

	/** compute dot product between vector1 and a dense vector
	 *
	 * @param vec_idx1 index of first vector
	 * @param vec2 dense vector
	 */
	float64_t
	dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const override;

	/** add vector 1 multiplied with alpha to dense vector2
	 *
	 * @param alpha scalar alpha
	 * @param vec_idx1 index of first vector
	 * @param vec2 pointer to real valued vector
	 * @param vec2_len length of real valued vector
	 * @param abs_val if true add the absolute value
	 */
	void add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val=false) const override;

	/** get number of non-zero features in vector
	 *
	 * (in case accurate estimates are too expensive overestimating is OK)
	 *
	 * @param num which vector
	 * @return number of sparse features in vector
	 */
	int32_t get_nnz_features_for_vector(int32_t num) const override;

	/** iterate over the non-zero features
	 *
	 * call get_feature_iterator first, followed by get_next_feature and
	 * free_feature_iterator to cleanup
	 * NOT IMPLEMENTED
	 *
	 * @param vector_index the index of the vector over whose components to
	 *			iterate over
	 * @return feature iterator (to be passed to get_next_feature)
	 */
	void* get_feature_iterator(int32_t vector_index) override;

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
	bool get_next_feature(int32_t& index, float64_t& value, void* iterator) override;

	/** clean up iterator
	 * call this function with the iterator returned by get_feature_iterator
	 * NOT IMPLEMENTED
	 *
	 * @param iterator as returned by get_feature_iterator
	 */
	void free_feature_iterator(void* iterator) override;

	/** set the document collection to work on
	 *
	 * @param docs the document collection
	 */
	void set_doc_collection(std::shared_ptr<StringFeatures<char>> docs);

	const char* get_name() const override;

	/** duplicate feature object
	 *
	 * @return feature object
	 */
	std::shared_ptr<Features> duplicate() const override;

	/** get feature type
	 *
	 * @return templated feature type
	 */
	EFeatureType get_feature_type() const override;

	/** get feature class
	 *
	 * @return feature class DENSE
	 */
	EFeatureClass get_feature_class() const override;

	/** get number of feature vectors
	 *
	 * @return number of feature vectors
	 */
	int32_t get_num_vectors() const override;

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
	void init();

protected:
	/** the document collection*/
	std::shared_ptr<StringFeatures<char>> doc_collection;

	/** number of bits of hash */
	int32_t num_bits = 0;

	/** tokenizer */
	std::shared_ptr<Tokenizer> tokenizer;

	/** if should normalize the dot product results */
	bool should_normalize = true;

	/** n for ngrams for quadratic features */
	int32_t ngrams = 1;

	/** tokens to skip when combining tokens */
	int32_t tokens_to_skip = 0;
};
}

#endif
