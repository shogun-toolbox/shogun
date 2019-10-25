/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Vladislav Horbatiuk, Bjoern Esser,
 *          Sergey Lisitsyn
 */

#ifndef _DIRECTORDOTFEATURES_H___
#define _DIRECTORDOTFEATURES_H___

#include <shogun/lib/config.h>

#ifdef USE_SWIG_DIRECTORS
#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/features/DotFeatures.h>

namespace shogun
{
/** @brief DirectorDotFeatures that support dot products among other operations and can be overloaded in modular interfaces.
 */
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class DirectorDotFeatures : public DotFeatures
{
	public:

		/** constructor
		 *
		 * @param size cache size
		 */
		DirectorDotFeatures(int32_t size=0) : DotFeatures(size)
		{
		}

		virtual ~DirectorDotFeatures() { }

		/** get number of examples/vectors, possibly corresponding to the current subset
		 *
		 * abstract base method
		 *
		 * @return number of examples/vectors (possibly of subset, if implemented)
		 */
		virtual int32_t get_num_vectors() const
		{
			not_implemented(SOURCE_LOCATION);
			return 0;
		}


		/** obtain the dimensionality of the feature space
		 *
		 * (not mix this up with the dimensionality of the input space, usually
		 * obtained via get_num_features())
		 *
		 * @return dimensionality
		 */
		virtual int32_t get_dim_feature_space() const
		{
			not_implemented(SOURCE_LOCATION);
			return 0;
		}

		/** compute dot product between vector1 and vector2,
		 * appointed by their indices
		 *
		 * @param vec_idx1 index of first vector
		 * @param df DotFeatures (of same kind) to compute dot product with
		 * @param vec_idx2 index of second vector
		 */
		virtual float64_t dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df, int32_t vec_idx2) const
		{
			not_implemented(SOURCE_LOCATION);
			return 0;
		}

		/** compute dot product between vector1 and a dense vector
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec2 dense vector
		 */
		virtual float64_t dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const override
		{
			not_implemented(SOURCE_LOCATION);
			return 0;
		}

		/** add vector 1 multiplied with alpha to dense vector2
		 *
		 * @param alpha scalar alpha
		 * @param vec_idx1 index of first vector
		 * @param vec2 pointer to real valued vector
		 * @param vec2_len length of real valued vector
		 * @param abs_val if true add the absolute value
		 */
		virtual void add_to_dense_sgvec(float64_t alpha, int32_t vec_idx1, SGVector<float64_t> vec2, bool abs_val=false)
		{
			not_implemented(SOURCE_LOCATION);
		}

		/** add vector 1 multiplied with alpha to dense vector2
		 *
		 * @param alpha scalar alpha
		 * @param vec_idx1 index of first vector
		 * @param vec2 pointer to real valued vector
		 * @param vec2_len length of real valued vector
		 * @param abs_val if true add the absolute value
		 */
		virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val=false)
		{
			add_to_dense_sgvec(alpha, vec_idx1, SGVector<float64_t>(vec2, vec2_len, false), abs_val);
		}

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
		virtual void dense_dot_range(float64_t* output, int32_t start, int32_t stop, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b)
		{
			DotFeatures::dense_dot_range(output, start, stop, alphas, vec, dim, b);
		}

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
				float64_t* output, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b)
		{
			DotFeatures::dense_dot_range_subset(sub_index, num, output, alphas, vec, dim, b);
		}

		/** get number of non-zero features in vector
		 *
		 * (in case accurate estimates are too expensive overestimating is OK)
		 *
		 * @param num which vector
		 * @return number of sparse features in vector
		 */
		virtual int32_t get_nnz_features_for_vector(int32_t num) const
		{
			not_implemented(SOURCE_LOCATION);
			return 0;
		}

		/** iterate over the non-zero features
		 *
		 * call get_feature_iterator first, followed by get_next_feature and
		 * free_feature_iterator to cleanup
		 *
		 * @param vector_index the index of the vector over whose components to
		 *			iterate over
		 * @return feature iterator (to be passed to get_next_feature)
		 */
		virtual void* get_feature_iterator(int32_t vector_index)
		{
			not_implemented(SOURCE_LOCATION);
			return NULL;
		}

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
		virtual bool get_next_feature(int32_t& index, float64_t& value, void* iterator)
		{
			not_implemented(SOURCE_LOCATION);
			return false;
		}

		/** clean up iterator
		 * call this function with the iterator returned by get_feature_iterator
		 *
		 * @param iterator as returned by get_feature_iterator
		 */
		virtual void free_feature_iterator(void* iterator)
		{
			not_implemented(SOURCE_LOCATION);
		}

		/** get mean
		 *
		 * @return mean returned
		 */
		virtual SGVector<float64_t> get_mean()
		{
			return DotFeatures::get_mean();
		}

		/** get covariance
		 *
		 * @return covariance
		 */
		virtual SGMatrix<float64_t> get_cov()
		{
			return DotFeatures::get_cov();
		}

		/** get feature type
		 *
		 * abstract base method
		 *
		 * @return templated feature type
		 */
		virtual EFeatureType get_feature_type() const
		{
			return F_ANY;
		}

		/** duplicate feature object
		 *
		 * abstract base method
		 *
		 * @return feature object
		 */
		virtual ::std::shared_ptr<Features> duplicate() const
		{
			not_implemented(SOURCE_LOCATION);
			return NULL;
		}

		/** get feature class
		 *
		 * abstract base method
		 *
		 * @return feature class like STRING, SIMPLE, SPARSE...
		 */
		virtual EFeatureClass get_feature_class() const
		{
			return C_DIRECTOR_DOT;
		}

		/** add preprocessor
		 *
		 * @param p preprocessor to set
		 */
		virtual void add_preprocessor(std::shared_ptr<Preprocessor> p)
		{
			Features::add_preprocessor(p);
		}

		/** delete preprocessor from list
		 * caller has to clean up returned preproc
		 *
		 * @param num index of preprocessor in list
		 */
		virtual void del_preprocessor(int32_t num)
		{
			Features::del_preprocessor(num);
		}

		/** in case there is a feature matrix allow for reshaping
		 *
		 * NOT IMPLEMENTED!
		 *
		 * @param num_features new number of features
		 * @param num_vectors new number of vectors
		 * @return if reshaping was successful
		 */
		virtual bool reshape(int32_t num_features, int32_t num_vectors)
		{
			not_implemented(SOURCE_LOCATION);
			return false;
		}

		/** load features from file
		 *
		 * @param loader File object via which data shall be loaded
		 */
		virtual void load(std::shared_ptr<File> loader)
		{
			Features::load(loader);
		}

		/** save features to file
		 *
		 * @param writer File object via which data shall be saved
		 */
		virtual void save(std::shared_ptr<File> writer)
		{
			Features::save(writer);
		}

		/** adds a subset of indices on top of the current subsets (possibly
		 * subset o subset. Calls subset_changed_post() afterwards
		 *
		 * @param subset subset of indices to add
		 * */
		virtual void add_subset(SGVector<index_t> subset)
		{
			Features::add_subset(subset);
		}

		/** removes that last added subset from subset stack, if existing
		 * Calls subset_changed_post() afterwards */
		virtual void remove_subset()
		{
			Features::remove_subset();
		}

		/** removes all subsets
		 * Calls subset_changed_post() afterwards */
		virtual void remove_all_subsets()
		{
			Features::remove_all_subsets();
		}

		/** method may be overwritten to update things that depend on subset */
		virtual void subset_changed_post()
		{
			Features::subset_changed_post();
		}

		/** Creates a new Features instance containing copies of the elements
		 * which are specified by the provided indices.
		 *
		 * This method is needed for a KernelMachine to store its model data.
		 * NOT IMPLEMENTED!
		 *
		 * @param indices indices of feature elements to copy
		 * @return new Features instance with copies of feature data
		 */
		virtual std::shared_ptr<Features> copy_subset(SGVector<index_t> indices) const
		{
			return Features::copy_subset(indices);
		}

		/** @return object name */
		virtual const char* get_name() const { return "DirectorDotFeatures"; }
};
}
#endif // USE_SWIG_DIRECTORS
#endif // _DIRECTORDOTFEATURES_H___
