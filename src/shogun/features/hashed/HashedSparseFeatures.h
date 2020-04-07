/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#ifndef _HASHED_SPARSEFEATURES_H__
#define _HASHED_SPARSEFEATURES_H__

#include <shogun/lib/config.h>

#include <shogun/features/SparseFeatures.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/lib/SGSparseVector.h>

namespace shogun
{
template <class ST> class SparseFeatures;
template <class ST> class SGSparseVector;
class DotFeatures;

/** @brief This class is identical to the DenseFeatures class
 * except that it hashes each dimension to a new feature space.
 */
template <class ST> class HashedSparseFeatures  : public DotFeatures
{
public:

	/** constructor
	 *
	 * @param size cache size
	 * @param use_quadr whether to use quadratic features or not
	 * @param keep_lin_terms whether to maintain the linear terms in the computations
	 */
	HashedSparseFeatures(int32_t size=0, bool use_quadr = false, bool keep_lin_terms = true);

	/** constructor
	 *
	 * @param feats	the sparse features to use as a base
	 * @param d new feature space dimension
	 * @param use_quadr whether to use quadratic features or not
	 * @param keep_lin_terms whether to maintain the linear terms in the computations
	 */
	HashedSparseFeatures(std::shared_ptr<SparseFeatures<ST>> feats, int32_t d, bool use_quadr = false,
			bool keep_lin_terms = true);

	/** constructor
	 *
	 * @param matrix feature matrix
	 * @param d new feature space dimension
	 * @param use_quadr whether to use quadratic features or not
	 * @param keep_lin_terms whether to maintain the linear terms in the computations
	 */
	HashedSparseFeatures(SGSparseMatrix<ST> matrix, int32_t d, bool use_quadr = false,
			bool keep_lin_terms = true);

	/** constructor loading features from file
	 *
	 * @param loader File object via which to load data
	 * @param d new feature space dimension
	 * @param use_quadr whether to use quadratic features or not
	 * @param keep_lin_terms whether to maintain the linear terms in the computations
	 */
	HashedSparseFeatures(std::shared_ptr<File> loader, int32_t d, bool use_quadr = false,
			bool keep_lin_terms = true);

	/** copy constructor */
	HashedSparseFeatures(const HashedSparseFeatures & orig);

	/** duplicate */
	std::shared_ptr<Features> duplicate() const override;

	/** destructor */
	~HashedSparseFeatures() override;

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
	 * possible with subset
	 *
	 * @param vec_idx1 index of first vector
	 * @param df DotFeatures (of same kind) to compute dot product with
	 * @param vec_idx2 index of second vector
	 */
	float64_t dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df,
			int32_t vec_idx2) const override;

	/** compute dot product between vector1 and a dense vector
	 *
	 * possible with subset
	 *
	 * @param vec_idx1 index of first vector
	 * @param vec2 dense vector
	 */
	float64_t
	dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const override;

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
	void add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
			float64_t* vec2, int32_t vec2_len, bool abs_val = false) const override;

	/** get number of non-zero features in vector
	 *
	 * @param num which vector
	 * @return number of non-zero features in vector
	 */
	int32_t get_nnz_features_for_vector(int32_t num) const override;

	/** iterate over the non-zero features
	 *
	 * call get_feature_iterator first, followed by get_next_feature and
	 * free_feature_iterator to cleanup
	 *
	 * possible with subset
	 *
	 * @param vector_index the index of the vector over whose components to
	 *			iterate over
	 * @return feature iterator (to be passed to get_next_feature)
	 */
	void* get_feature_iterator(int32_t vector_index) override;

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
	bool get_next_feature(int32_t& index, float64_t& value,
			void* iterator) override;

	/** clean up iterator
	 * call this function with the iterator returned by get_first_feature
	 *
	 * @param iterator as returned by get_first_feature
	 */
	void free_feature_iterator(void* iterator) override;

	/** @return object name */
	const char* get_name() const override;

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

	/** Get the hashed representation of the specified vector
	 *
	 * @param vec_idx the index of the vector
	 */
	SGSparseVector<ST> get_hashed_feature_vector(int32_t vec_idx) const;

	/** Get the hashed representation of the given vector
	 *
	 * @param vec the vector to hash
	 * @param dim the dimension of the new feature space
	 * @param use_quadratic whether to use quadratic features or not
	 * @param keep_linear_terms whether to maintain the linear terms in the computations
	 * @return the hashed representation of the vector vec
	 */
	static SGSparseVector<ST> hash_vector(SGVector<ST> vec, int32_t dim,
		bool use_quadratic = false, bool keep_linear_terms = true);


	/** Get the hashed representation of the given sparse vector
	 *
	 * @param vec the vector to hash
	 * @param dim the dimension of the hashed target space
	 * @param use_quadratic whether to use quadratic features or not
	 * @param keep_linear_terms whether to maintain the linear terms in the computations
	 * @return the hashed representation of the vector vec
	 */
	static SGSparseVector<ST> hash_vector(SGSparseVector<ST> vec, int32_t dim,
		bool use_quadratic = false, bool keep_linear_terms = true);

private:
	void init(std::shared_ptr<SparseFeatures<ST>> feats, int32_t d, bool use_quadr, bool keep_lin_terms);

protected:

	/** sparse features */
	std::shared_ptr<SparseFeatures<ST>> sparse_feats;

	/** new feature space dimension */
	int32_t dim;

	/** use quadratic features */
	bool use_quadratic;

	/** keep linear terms */
	bool keep_linear_terms;
};
}

#endif // _HASHED_SPARSEFEATURES_H__
