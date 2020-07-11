/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Heiko Strathmann,
 *          Vladislav Horbatiuk, Yuyu Zhang, Bjoern Esser, Sergey Lisitsyn
 */

#ifndef _WDFEATURES_H___
#define _WDFEATURES_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/StringFeatures.h>

namespace shogun
{
template <class ST> class StringFeatures;

/** @brief Features that compute the Weighted Degreee Kernel feature space
 * explicitly.
 *
 * \sa CWeightedDegreeStringKernel
 */
class WDFeatures : public DotFeatures
{
	public:
		/** defualt constructor  */
		WDFeatures();

		/** constructor
		 *
		 * @param str stringfeatures (of bytes)
		 * @param order of wd kernel
		 * @param from_order use first order weights from higher order weighting
		 */
		WDFeatures(const std::shared_ptr<StringFeatures<uint8_t>>& str, int32_t order, int32_t from_order);

		/** copy constructor */
		WDFeatures(const WDFeatures & orig);

		/** destructor */
		~WDFeatures() override;

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
		void add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
				float64_t* vec2, int32_t vec2_len, bool abs_val=false) const override;

		/** get number of non-zero features in vector
		 *
		 * @param num which vector
		 * @return number of non-zero features in vector
		 */
		int32_t get_nnz_features_for_vector(int32_t num) const override;

		#ifndef DOXYGEN_SHOULD_SKIP_THIS
		/** iterator for weighted spectrum features */
		struct wd_feature_iterator
		{
			/** pointer to feature vector */
			uint8_t* vec;
			/** index of vector */
			int32_t vidx;
			/** length of vector */
			int32_t vlen;
			/** if we need to free the vector*/
			bool vfree;

			/** @name Internal Parameters
			 * parameters of interal feature gen loop
			 */
			//@{
			int32_t lim;
			int32_t* val;
			int32_t asize;
			int32_t asizem1;
			int32_t offs;
			int32_t k;
			int32_t i;
			int32_t o;
			//@}
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
		bool get_next_feature(int32_t& index, float64_t& value, void* iterator) override;

		/** clean up iterator
		 * call this function with the iterator returned by get_first_feature
		 *
		 * @param iterator as returned by get_first_feature
		 */
		void free_feature_iterator(void* iterator) override;

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
		 * @return feature class
		 */
		EFeatureClass get_feature_class() const override;

		int32_t get_num_vectors() const override;

		/** set normalization constant
		 * @param n n=0 means automagic */
		void set_normalization_const(float64_t n=0);

		/** get normalization constant */
		float64_t get_normalization_const();

		/** @return object name */
		const char* get_name() const override { return "WDFeatures"; }

		/** set wd weights
		 *
		 * @param weights new weights
		 */
		void set_wd_weights(SGVector<float64_t> weights);

		/** create wd kernel weighting heuristic */
		void set_wd_weights();

	protected:
		/** stringfeatures the wdfeatures are based on*/
		std::shared_ptr<StringFeatures<uint8_t>> strings;

		/** degree */
		int32_t degree;
		/** from degree */
		int32_t from_degree;
		/** length of string in vector */
		int32_t string_length;
		/** number of strings */
		int32_t num_strings;
		/** size of alphabet */
		int32_t alphabet_size;
		/** w dim */
		int32_t w_dim;
		/** wd weights */
		float64_t* wd_weights;

		/** normalization const */
		float64_t normalization_const;

};
}
#endif // _WDFEATURES_H___
