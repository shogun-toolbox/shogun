/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn,
 *          Vladislav Horbatiuk, Evgeniy Andreev, Yuyu Zhang, Bjoern Esser
 */
#ifndef _POLYFEATURES__H__
#define _POLYFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/DenseFeatures.h>


namespace shogun
{
/** @brief implement DotFeatures for the polynomial kernel
 *
 * see DotFeatures for further discription
 *
 */
class PolyFeatures : public DotFeatures
{
	public:
		/** default constructor  */
		PolyFeatures();

		/** constructor
		 *
		 * @param feat real features
		 * @param degree degree of the polynomial kernel
		 * @param normalize normalize kernel
		 */
		PolyFeatures(const std::shared_ptr<DenseFeatures<float64_t>>& feat, int32_t degree, bool normalize);

		~PolyFeatures() override;

		/** copy constructor
		 *
		 * not implemented!
		 *
		 * @param orig original PolyFeature
		 */
		PolyFeatures(const PolyFeatures & orig);

		/** get dimensions of feature space
		 *
		 * @return dimensions of feature space
		 */
		int32_t get_dim_feature_space() const override;

		/** get number of non-zero features in vector
		 *
		 * @param num index of vector
		 * @return number of non-zero features in vector
		 */
		int32_t get_nnz_features_for_vector(int32_t num) const override;

		/** get feature type
		 *
		 * @return feature type
		 */
		EFeatureType get_feature_type() const override;

		/** get feature class
		 *
		 * @return feature class
		 */
		EFeatureClass get_feature_class() const override;

		/** get number of vectors
		 *
		 * @return number of vectors
		 */
		int32_t get_num_vectors() const override;

		/** compute dot product between vector1 and vector2,
		 *  appointed by their indices
		 *
		 * @param vec_idx1 index of first vector
		 * @param df DotFeatures (of same kind) to compute dot product with
		 * @param vec_idx2 index of second vector
		 */
		float64_t dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df, int32_t vec_idx2) const override;

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		std::shared_ptr<Features> duplicate() const override;

		/**
		 *
		 * @return name of class
		 */
		const char* get_name() const override { return "PolyFeatures"; }

		/** compute dot product between vector1 and a dense vector
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec2 dense vector
		 */
		float64_t
		dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const override;

		/** compute alpha*x+vec2
		 *
		 * @param alpha alpha
		 * @param vec_idx1 index of first vector x
		 * @param vec2 vec2
		 * @param vec2_len length of vec2
		 * @param abs_val if true add the absolute value
		 */
		void add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val) const override;

		#ifndef DOXYGEN_SHOULD_SKIP_THIS
		/** iterator for weighted spectrum features */
		struct poly_feature_iterator
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
		void* get_feature_iterator(int32_t vector_index) override;

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
		bool get_next_feature(int32_t& index, float64_t& value,
				void* iterator) override;

		/** clean up iterator
		 * call this function with the iterator returned by get_first_feature
		 *
		 * @param iterator as returned by get_first_feature
		 */
		void free_feature_iterator(void* iterator) override;

	protected:

		/** store the norm of each training example */
		void store_normalization_values();

		/** caller function for the recursive function enumerate_multi_index */
		void store_multi_index();

		/** recursive function enumerating all multi-indices that sum
		 *  up to the degree of the polynomial kernel */
		void enumerate_multi_index(const int32_t feat_idx, uint16_t** index,
				uint16_t* exponents, const int32_t degree);
		/** function calculating the multinomial coefficients for all
		 *  multi indices */
		void store_multinomial_coefficients();

		/** simple recursive implementation of binomial coefficient
		 *  which is very efficient if k is small, otherwise it calls
		 *  a more sophisticated implementation */
		int32_t bico2(int32_t n, int32_t k);

		/** efficient implementation for the binomial coefficient function
		 *  for larger values of k*/
		int32_t  bico(int32_t n, int32_t k);

		/** recursion to calculate the dimensions of the feature space:
		 *  A(N, D)= sum_d=0^D A(N-1, d)
		 *  A(1, D)==1
		 *  A(N, 0)==1
		 *  where N is the dimensionality of the input space
		 *  and D is the degree */
		int32_t calc_feature_space_dimensions(int32_t N, int32_t D);

		/** calculate the multinomial coefficient */
		int32_t multinomialcoef(int32_t* exps, int32_t len);

		/** efficient implementation of the ln(gamma(x)) function*/
		float64_t gammln(float64_t xx);

		/** implementation of the ln(x!) function*/
		float64_t factln(int32_t n);

	protected:

		/** features in original space*/
		std::shared_ptr<DenseFeatures<float64_t>> m_feat;
		/** degree of the polynomial kernel */
		int32_t m_degree;
		/** normalize */
		bool m_normalize;
		/** dimensions of the input space */
		int32_t m_input_dimensions;
		/** dimensions of the feature space of the polynomial kernel */
		int32_t m_output_dimensions;
		/** flattened matrix of all multi indices that
		 *  sum do the degree of the polynomial kernel */
		uint16_t* m_multi_index;
		/** multinomial coefficients for all multi-indices */
		float64_t* m_multinomial_coefficients;
		/**store norm of each training example */
		float32_t* m_normalization_values;
	private:
		index_t multi_index_length;
		index_t multinomial_coefficients_length;
		index_t normalization_values_length;

		/** Register all parameters */
		void register_parameters();
};
}
#endif // _POLYFEATURES__H__
