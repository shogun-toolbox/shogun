/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CCOMBINEDFEATURES__H__
#define _CCOMBINEDFEATURES__H__

#include <shogun/features/Features.h>
#include <shogun/lib/List.h>

namespace shogun
{
class CFeatures;
class CList;
class CListElement;

/** @brief The class CombinedFeatures is used to combine a number of of feature objects
 * into a single CombinedFeatures object.
 *
 * It keeps pointers to the added sub-features and is especially useful to
 * combine kernels working on different domains (c.f. CCombinedKernel) and to
 * combine kernels looking at independent features.
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
		inline virtual EFeatureType get_feature_type() const
		{
			return F_UNKNOWN;
		}

		/** get feature class
		 *
		 * @return feature class SIMPLE
		 */
		inline virtual EFeatureClass get_feature_class() const
		{
			return C_COMBINED;
		}

		/** get number of feature vectors
		 *
		 * @return number of feature vectors
		 */
		inline virtual int32_t get_num_vectors() const
		{
			return num_vec;
		}

		/** get memory footprint of one feature
		 *
		 * @return memory footprint of one feature
		 */
		virtual int32_t get_size() const;

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

		/** get first feature object
		 *
		 * @param current list of features
		 * @return first feature object
		 */
		CFeatures* get_first_feature_obj(CListElement*& current);

		/** get next feature object
		 *
		 * @return next feature object
		 */
		CFeatures* get_next_feature_obj();

		/** get next feature object
		 *
		 * @param current list of features
		 * @return next feature object
		 */
		CFeatures* get_next_feature_obj(CListElement*& current);

		/** get last feature object
		 *
		 * @return last feature object
		 */
		CFeatures* get_last_feature_obj();

		/** insert feature object
		 *
		 * @param obj feature object to insert
		 * @return if inserting was successful
		 */
		bool insert_feature_obj(CFeatures* obj);

		/** append feature object
		 *
		 * @param obj feature object to append
		 * @return if appending was successful
		 */
		bool append_feature_obj(CFeatures* obj);

		/** delete feature object
		 *
		 * @return if deleting was successful
		 */
		bool delete_feature_obj();

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

		/** @return object name */
		inline virtual const char* get_name() const { return "CombinedFeatures"; }

	private:
		void init();

	protected:
		/** feature list */
		CList* feature_list;

		/** number of vectors
		 * must match between sub features
		 */
		int32_t num_vec;
};
}
#endif
