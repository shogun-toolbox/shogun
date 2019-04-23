/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Evangelos Anagnostopoulos,
 *          Vladislav Horbatiuk, Yuyu Zhang, Evgeniy Andreev, Bjoern Esser
 */

#ifndef _CCOMBINEDFEATURES__H__
#define _CCOMBINEDFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/features/Features.h>
#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{
class Features;
class DynamicObjectArray;

/** @brief The class CombinedFeatures is used to combine a number of of feature objects
 * into a single CombinedFeatures object.
 *
 * It keeps pointers to the added sub-features and is especially useful to
 * combine kernels working on different domains (c.f. CombinedKernel) and to
 * combine kernels looking at independent features.
 *
 * Subsets are supported: All actions will just be given through to all
 * sub-features. Only once per sub-feature instance, i.e. if there are two
 * sub-features that are the same instance, the subset action will only be
 * performed once.
 */
class CombinedFeatures : public Features
{
	public:
		/** default constructor */
		CombinedFeatures();
		/** copy constructor */
		CombinedFeatures(const CombinedFeatures& orig);

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual std::shared_ptr<Features> duplicate() const;

		/** destructor */
		virtual ~CombinedFeatures();

		/** get feature type
		 *
		 * @return feature type UNKNOWN
		 */
		virtual EFeatureType get_feature_type() const
		{
			return F_UNKNOWN;
		}

		/** get feature class
		 *
		 * @return feature class SIMPLE
		 */
		virtual EFeatureClass get_feature_class() const
		{
			return C_COMBINED;
		}

		/** get number of feature vectors
		 *
		 * @return number of feature vectors
		 */
		virtual int32_t get_num_vectors() const
		{
			return m_subset_stack->has_subsets()
					? m_subset_stack->get_size() : num_vec;
		}

		/** list feature objects */
		void list_feature_objs() const;

		/** check feature object compatibility
		 *
		 * @param comb_feat feature to check for compatibility
		 * @return if feature is compatible
		 */
		bool check_feature_obj_compatibility(std::shared_ptr<CombinedFeatures> comb_feat);

		/** get first feature object
		 *
		 * @return first feature object
		 */
		std::shared_ptr<Features> get_first_feature_obj() const;

		/** get feature object at index idx
		*
		* @param idx index of feature object
		* @return the feature object at index idx
		*/
		std::shared_ptr<Features> get_feature_obj(int32_t idx) const;

		/** get last feature object
		 *
		 * @return last feature object
		 */
		std::shared_ptr<Features> get_last_feature_obj() const;

		/** insert feature object at index idx
		 *  Important, idx must be < num_feature_obj
		 *
		 * @param obj feature object to insert
		 * @param idx the index where to insert the feature object
		 * @return if inserting was successful
		 */
		bool insert_feature_obj(std::shared_ptr<Features> obj, int32_t idx);

		/** append feature object to the end of this CombinedFeatures object array
		 *
		 * @param obj feature object to append
		 * @return if appending was successful
		 */
		bool append_feature_obj(std::shared_ptr<Features> obj);

		/** delete feature object at position idx
		 *
		 * @param idx the index of the feature object to delete
		 * @return if deleting was successful
		 */
		bool delete_feature_obj(int32_t idx);

		/** get number of feature objects
		 *
		 * @return number of feature objects
		 */
		int32_t get_num_feature_obj() const;

		/** Takes another feature instance and returns a new instance which is
		 * a concatenation of a copy if this instace's data and the given
		 * instance's data. Note that the feature types have to be equal.
		 *
		 * In this case, all sub features are merged
		 *
		 * @param other feature object to append
		 * @return new feature object which contains copy of data of this
		 * instance and of given one
		 */
		std::shared_ptr<Features> create_merged_copy(std::shared_ptr<Features> other) const;

		/** adds a subset of indices on top of the current subsets (possibly
		 * subset o subset. Calls subset_changed_post() afterwards.
		 * Adds the subset to all sub-features
		 *
		 * @param subset subset of indices to add
		 * */
		virtual void add_subset(SGVector<index_t> subset);

		/** removes that last added subset from subset stack, if existing
		 * Calls subset_changed_post() afterwards
		 *
		 * Removes the subset from all sub-features
		 * */
		virtual void remove_subset();

		/** removes all subsets
		 * Calls subset_changed_post() afterwards
		 *
		 * Removes all subsets of all sub-features
		 * */
		virtual void remove_all_subsets();

		/** Creates a new Features instance containing copies of the elements
		 * which are specified by the provided indices.
		 * Simply creates a combined features instance where all sub-features
		 * are the results of their copy_subset calls
		 *
		 * @param indices indices of feature elements to copy
		 * @return new Features instance with copies of feature data
		 */
		virtual std::shared_ptr<Features> copy_subset(SGVector<index_t> indices) const;

		/** @return object name */
		virtual const char* get_name() const { return "CombinedFeatures"; }

	private:
		void init();

	protected:
		/** feature array */
		std::vector<std::shared_ptr<Features>> feature_array;

		/** number of vectors
		 * must match between sub features
		 */
		int32_t num_vec;
};
}
#endif
