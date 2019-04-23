/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Vladislav Horbatiuk,
 *          Evgeniy Andreev, Yuyu Zhang, Evan Shelhamer, Bjoern Esser,
 *          Sergey Lisitsyn
 */
#ifndef _SPARSEPOLYFEATURES__H__
#define _SPARSEPOLYFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/SparseFeatures.h>

namespace shogun
{
/** @brief implement DotFeatures for the polynomial kernel
 *
 * see DotFeatures for further discription
 *
 */
class SparsePolyFeatures : public DotFeatures
{
	public:
		/** default constructor  */
		SparsePolyFeatures();

		/** constructor
		 *
		 * @param feat real features
		 * @param degree degree of the polynomial kernel
		 *					(only degree 2 & 3 are supported)
		 * @param normalize normalize kernel
		 * @param hash_bits number of bits in hashd feature space
		 */
		SparsePolyFeatures(std::shared_ptr<SparseFeatures<float64_t>> feat, int32_t degree, bool normalize, int32_t hash_bits);

		virtual ~SparsePolyFeatures();

		/** copy constructor
		 *
		 * not implemented!
		 *
		 * @param orig original PolyFeature
		 */
		SparsePolyFeatures(const SparsePolyFeatures & orig);

		/** get dimensions of feature space
		 *
		 * @return dimensions of feature space
		 */
		virtual int32_t get_dim_feature_space() const;

		/** get number of non-zero features in vector
		 *
		 * @param num index of vector
		 * @return number of non-zero features in vector
		 */
		virtual int32_t get_nnz_features_for_vector(int32_t num) const;

		/** get feature type
		 *
		 * @return feature type
		 */
		virtual EFeatureType get_feature_type() const;

		/** get feature class
		 *
		 * @return feature class
		 */
		virtual EFeatureClass get_feature_class() const;

		/** get number of vectors
		 *
		 * @return number of vectors
		 */
		virtual int32_t get_num_vectors() const;

		/** compute dot product between vector1 and vector2,
		 *  appointed by their indices
		 *
		 * @param vec_idx1 index of first vector
		 * @param df DotFeatures (of same kind) to compute dot product with
		 * @param vec_idx2 index of second vector
		 */
		virtual float64_t dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df, int32_t vec_idx2) const;

		#ifndef DOXYGEN_SHOULD_SKIP_THIS
		/** iterator for weighted spectrum features */
		struct sparse_poly_feature_iterator
		{
			/** pointer to feature vector */
			uint16_t* vec;
			/** index of vector */
			int32_t vidx;
			/** length of vector */
			int32_t vlen;
			/** if we need to free the vector*/
			bool vfree;

			/** feature index */
			int32_t index;
		};
		#endif

		/** iterate over the non-zero features
		 *
		 * call get_feature_iterator first, followed by get_next_feature and
		 * free_feature_iterator to cleanup
		 *
		 * @param vector_index the index of the vector over whose components to
		 *			iterate over
		 * @return feature iterator (to be passed to get_next_feature)
		 */
		virtual void* get_feature_iterator(int32_t vector_index);

		/** iterate over the non-zero features
		 *
		 * call this function with the iterator returned by get_first_feature
		 * and call free_feature_iterator to cleanup
		 *
		 * @param index is returned by reference (-1 when not available)
		 * @param value is returned by reference
		 * @param iterator as returned by get_first_feature
		 * @return true if a new non-zero feature got returned
		 */
		bool get_next_feature(int32_t& index, float64_t& value, void* iterator);

		/** clean up iterator
		 * call this function with the iterator returned by get_first_feature
		 *
		 * @param iterator as returned by get_first_feature
		 */
		void free_feature_iterator(void* iterator);

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		std::shared_ptr<Features> duplicate() const;

		/**
		 *
		 * @return name of class
		 */
		virtual const char* get_name() const { return "SparsePolyFeatures"; }

		/** compute dot product between vector1 and a dense vector
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec2 dense vector
		 */
		virtual float64_t
		dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const override;

		/** compute alpha*x+vec2
		 *
		 * @param alpha alpha
		 * @param vec_idx1 index of first vector x
		 * @param vec2 vec2
		 * @param vec2_len length of vec2
		 * @param abs_val if true add the absolute value
		 */
		virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val=false) const;

	protected:
		/** store the norm of each training example */
		void store_normalization_values();

	protected:
		/** features in original space*/
		std::shared_ptr<SparseFeatures<float64_t>> m_feat;
		/** degree of the polynomial kernel */
		int32_t m_degree;
		/** normalize */
		bool m_normalize;
		/** dimensions of the input space */
		int32_t m_input_dimensions;
		/** dimensions of the feature space of the polynomial kernel */
		int32_t m_output_dimensions;
		/**store norm of each training example */
		float64_t* m_normalization_values;
		/** mask */
		uint32_t mask;
		/** number of bits in hash */
		int32_t m_hash_bits;
	private:
		/**Initialize parameters for serialization*/
		void init();

	private:
		/**length of norm for each traning example*/
		int32_t m_normalization_values_len;
};
}
#endif // _SPARSEPOLYFEATURES__H__
