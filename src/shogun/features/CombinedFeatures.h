/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CCOMBINEDFEATURES__H__
#define _CCOMBINEDFEATURES__H__

#include <shogun/lib/config.h>
#include <shogun/features/Features.h>
#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{
class CFeatures;
class CDynamicObjectArray;

/** @brief The class CombinedFeatures is used to combine a number of of feature objects
 * into a single CombinedFeatures object.
 *
 * It keeps pointers to the added sub-features and is especially useful to
 * combine kernels working on different domains (c.f. CCombinedKernel) and to
 * combine kernels looking at independent features.
 *
 * Subsets are supported: All actions will just be given through to all
 * sub-features. Only once per sub-feature instance, i.e. if there are two
 * sub-features that are the same instance, the subset action will only be
 * performed once.
 */
class CCombinedFeatures : public CFeatures
{
	public:
		/** default constructor */
		CCombinedFeatures();
		/** copy constructor */
		CCombinedFeatures(const CCombinedFeatures& orig);

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const;

		/** destructor */
		virtual ~CCombinedFeatures();

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
		void list_feature_objs();

		/** check feature object compatibility
		 *
		 * @param comb_feat feature to check for compatibility
		 * @return if feature is compatible
		 */
		bool check_feature_obj_compatibility(CCombinedFeatures* comb_feat);

		/** get first feature object
		 *
		 * @return first feature object
		 */
		CFeatures* get_first_feature_obj();

		/** get feature object at index idx
		*
		* @param idx index of feature object
		* @return the feature object at index idx
		*/
		CFeatures* get_feature_obj(int32_t idx);

		/** get last feature object
		 *
		 * @return last feature object
		 */
		CFeatures* get_last_feature_obj();

		/** insert feature object at index idx
		 *  Important, idx must be < num_feature_obj
		 *
		 * @param obj feature object to insert
		 * @param idx the index where to insert the feature object
		 * @return if inserting was successful
		 */
		bool insert_feature_obj(CFeatures* obj, int32_t idx);

		/** append feature object to the end of this CombinedFeatures object array
		 *
		 * @param obj feature object to append
		 * @return if appending was successful
		 */
		bool append_feature_obj(CFeatures* obj);

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
		int32_t get_num_feature_obj();

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
		CFeatures* create_merged_copy(CFeatures* other);

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

		/** Creates a new CFeatures instance containing copies of the elements
		 * which are specified by the provided indices.
		 * Simply creates a combined features instance where all sub-features
		 * are the results of their copy_subset calls
		 *
		 * @param indices indices of feature elements to copy
		 * @return new CFeatures instance with copies of feature data
		 */
		virtual CFeatures* copy_subset(SGVector<index_t> indices);

		/** @return object name */
		virtual const char* get_name() const { return "CombinedFeatures"; }

	private:
		void init();

	protected:
		/** feature array */
		CDynamicObjectArray* feature_array;

		/** number of vectors
		 * must match between sub features
		 */
		int32_t num_vec;
};
}
#endif
