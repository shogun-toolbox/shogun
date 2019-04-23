/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Michele Mazzoni, Evgeniy Andreev,
 *          Fernando Iglesias, Yuyu Zhang, Heiko Strathmann, Thoralf Klein,
 *          Evan Shelhamer, Bjoern Esser, Alesis Novik, Giovanni De Toni
 */

#ifndef _DOTFEATURES_H___
#define _DOTFEATURES_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{
/** @brief Features that support dot products among other operations.
 *
 * DotFeatures support the following operations:
 *
 * - a way to obtain the dimensionality of the feature space, i.e. \f$\mbox{dim}({\cal X})\f$
 *
 * - dot product between feature vectors:
 *
 *   \f[r = {\bf x} \cdot {\bf x'}\f]
 *
 * - dot product between feature vector and a dense vector \f${\bf z}\f$:
 *
 *   \f[r = {\bf x} \cdot {\bf z}\f]
 *
 * - multiplication with a scalar \f$\alpha\f$ and addition to a dense vector \f${\bf z}\f$:
 *
 *   \f[ {\bf z'} = \alpha {\bf x} + {\bf z} \f]
 *
 * - iteration over all (potentially) non-zero features of \f${\bf x}\f$
 *
 */
class DotFeatures : public Features
{
	public:

		/** constructor
		 *
		 * @param size cache size
		 */
		DotFeatures(int32_t size=0);

		/** copy constructor */
		DotFeatures(const DotFeatures & orig);

		/** constructor
		 *
		 * @param loader File object via which to load data
		 */
		DotFeatures(std::shared_ptr<File> loader);

		virtual ~DotFeatures() { }

		/** obtain the dimensionality of the feature space
		 *
		 * (not mix this up with the dimensionality of the input space, usually
		 * obtained via get_num_features())
		 *
		 * @return dimensionality
		 */
		virtual int32_t get_dim_feature_space() const=0;

		/** compute dot product between vector1 and vector2,
		 * appointed by their indices
		 *
		 * @param vec_idx1 index of first vector
		 * @param df DotFeatures (of same kind) to compute dot product with
		 * @param vec_idx2 index of second vector
		 */
		virtual float64_t dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df, int32_t vec_idx2) const = 0;

		/** compute dot product between vector1 and a dense vector
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec2 dense vector
		 */
		virtual float64_t
		dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const = 0;

		/** add vector 1 multiplied with alpha to dense vector2
		 *
		 * @param alpha scalar alpha
		 * @param vec_idx1 index of first vector
		 * @param vec2 pointer to real valued vector
		 * @param vec2_len length of real valued vector
		 * @param abs_val if true add the absolute value
		 */
		virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val=false) const = 0;

		/** Compute the dot product for a range of vectors. This function makes use of dense_dot
		 * alphas[i] * sparse[i]^T * w + b
		 *
		 * @param output result for the given vector range
		 * @param start start vector range from this idx
		 * @param stop stop vector range at this idx
		 * @param alphas scalars to multiply with, may be NULL
		 * @param vec dense vector to compute dot product with
		 * @param dim length of the dense vector
		 * @param b bias
		 *
		 * note that the result will be written to output[0...(stop-start-1)]
		 */
		virtual void dense_dot_range(float64_t* output, int32_t start, int32_t stop, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b) const;

		/** Compute the dot product for a subset of vectors. This function makes use of dense_dot
		 * alphas[i] * sparse[i]^T * w + b
		 *
		 * @param sub_index index for which to compute outputs
		 * @param num length of index
		 * @param output result for the given vector range
		 * @param alphas scalars to multiply with, may be NULL
		 * @param vec dense vector to compute dot product with
		 * @param dim length of the dense vector
		 * @param b bias
		 */
		virtual void dense_dot_range_subset(int32_t* sub_index, int32_t num,
				float64_t* output, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b) const;

		/** get number of non-zero features in vector
		 *
		 * (in case accurate estimates are too expensive overestimating is OK)
		 *
		 * @param num which vector
		 * @return number of sparse features in vector
		 */
		virtual int32_t get_nnz_features_for_vector(int32_t num) const=0;

		/** compute the feature matrix in feature space
		 *
		 * @return computed feature matrix
		 */
		SGMatrix<float64_t> get_computed_dot_feature_matrix() const;

		/** compute the feature vector in feature space
		 *
		 * @return computed feature vector
		 */
		SGVector<float64_t> get_computed_dot_feature_vector(int32_t num) const;

		/** iterate over the non-zero features
		 *
		 * call get_feature_iterator first, followed by get_next_feature and
		 * free_feature_iterator to cleanup
		 *
		 * @param vector_index the index of the vector over whose components to
		 *			iterate over
		 * @return feature iterator (to be passed to get_next_feature)
		 */
		virtual void* get_feature_iterator(int32_t vector_index)=0;

		/** iterate over the non-zero features
		 *
		 * call this function with the iterator returned by get_feature_iterator
		 * and call free_feature_iterator to cleanup
		 *
		 * @param index is returned by reference (-1 when not available)
		 * @param value is returned by reference
		 * @param iterator as returned by get_feature_iterator
		 * @return true if a new non-zero feature got returned
		 */
		virtual bool get_next_feature(int32_t& index, float64_t& value, void* iterator)=0;

		/** clean up iterator
		 * call this function with the iterator returned by get_feature_iterator
		 *
		 * @param iterator as returned by get_feature_iterator
		 */
		virtual void free_feature_iterator(void* iterator)=0;

		/** get mean
		 *
		 * @return mean returned
		 */
		virtual SGVector<float64_t> get_mean() const;

		/** get standard variance
		 *
		 * @param colwise if true calculates feature wise standard deviation,
		 * otherwise calculates the matrix standard deviation.
		 * @return Standard deviation of all feature vectors or of whole matrix
		 */
		virtual SGVector<float64_t> get_std(bool colwise = true) const;

		/** get mean of two CDotFeature objects
		 *
		 * @return mean returned
		 */
		static SGVector<float64_t>
		compute_mean(std::shared_ptr<DotFeatures> lhs, std::shared_ptr<DotFeatures> rhs);

		/** get covariance
		 *
		 * @param copy_data_for_speed if true, the method stores explicitly
		 * the centered data matrix and the covariance is calculated by matrix
		 * product of the centered data with its transpose, this make it
		 * possible to take advantage of multithreaded matrix product,
		 * this may not be possible if the data doesn't fit into memory,
		 * in such case set this parameter to false to compute iteratively
		 * the covariance matrix without storing the centered data.
		 * [default = true]
		 * @return covariance
		 */
		virtual SGMatrix<float64_t> get_cov(bool copy_data_for_speed = true) const;

		/** compute the covariance of two DotFeatures together
		 *
		 * @param copy_data_for_speed @see DotFeatures::get_cov
		 * @return covariance
		 */
		static SGMatrix<float64_t> compute_cov(
		    std::shared_ptr<DotFeatures> lhs, std::shared_ptr<DotFeatures> rhs,
		    bool copy_data_for_speed = true);

	private:
		void init();
};
}
#endif // _DOTFEATURES_H___
