/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Vladislav Horbatiuk,
 *          Evgeniy Andreev, Yuyu Zhang, Bjoern Esser, Sergey Lisitsyn
 */

#ifndef _EXPLICITSPECFEATURES_H___
#define _EXPLICITSPECFEATURES_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/StringFeatures.h>

namespace shogun
{
template <class ST> class StringFeatures;

/**@brief Features that compute the Spectrum Kernel feature space explicitly.
 *
 * \sa CCommWordStringKernel
 */
class ExplicitSpecFeatures : public DotFeatures
{
	public:
		/** default constructor */
		ExplicitSpecFeatures();

		/** constructor
		 *
		 * @param str stringfeatures (of words)
		 * @param normalize whether to use sqrtdiag normalization
		 */
		ExplicitSpecFeatures(std::shared_ptr<StringFeatures<uint16_t>> str, bool normalize=true);

		/** copy constructor */
		ExplicitSpecFeatures(const ExplicitSpecFeatures & orig);

		/** destructor */
		virtual ~ExplicitSpecFeatures();

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual std::shared_ptr<Features> duplicate() const;

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
		virtual float64_t dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df, int32_t vec_idx2) const;

		/** compute dot product between vector1 and a dense vector
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec2 dense vector
		 */
		virtual float64_t
		dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const override;

		/** add vector 1 multiplied with alpha to dense vector2
		 *
		 * @param alpha scalar alpha
		 * @param vec_idx1 index of first vector
		 * @param vec2 pointer to real valued vector
		 * @param vec2_len length of real valued vector
		 * @param abs_val if true add the absolute value
		 */
		virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
				float64_t* vec2, int32_t vec2_len, bool abs_val=false) const;

		#ifndef DOXYGEN_SHOULD_SKIP_THIS
		/** iterator for weighted spectrum features */
		struct explicit_spec_feature_iterator
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
		virtual bool get_next_feature(int32_t& index, float64_t& value, void* iterator);

		/** clean up iterator
		 * call this function with the iterator returned by get_first_feature
		 *
		 * @param iterator as returned by get_first_feature
		 */
		virtual void free_feature_iterator(void* iterator);

		/** get number of non-zero features in vector
		 *
		 * @param num which vector
		 * @return number of non-zero features in vector
		 */
		virtual int32_t get_nnz_features_for_vector(int32_t num) const;

		/** get feature type
		 *
		 * @return templated feature type
		 */
		virtual EFeatureType get_feature_type() const;

		/** get feature class
		 *
		 * @return feature class
		 */
		virtual EFeatureClass get_feature_class() const;

		/** get number of strings
		 *
		 * @return number of strings
		 */
		virtual int32_t get_num_vectors() const;

		/** @return object name */
		virtual const char* get_name() const { return "ExplicitSpecFeatures"; }

	protected:
		/** obtain the k-spectrum from a list of strings
		 *
		 * @param str the string feature object already in k-mer format
		 */
		void obtain_kmer_spectrum(std::shared_ptr<StringFeatures<uint16_t>> str);

		/** free kmer spectrum */
		void delete_kmer_spectrum();

	protected:
		/** use sqrtdiag normalization */
		bool use_normalization;
		/** number of strings */
		int32_t num_strings;
		/** size of alphabet */
		int32_t alphabet_size;

		/** size of k-mer spectrum*/
		int32_t spec_size;
		/** k-mer counts for all strings */
		float64_t** k_spectrum;
};
}
#endif // _EXPLICITSPECFEATURES_H___
