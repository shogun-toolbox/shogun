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

#ifndef _CFEATURES__H__
#define _CFEATURES__H__

#include "lib/common.h"
#include "lib/File.h"
#include "base/SGObject.h"
#include "preprocessor/Preprocessor.h"
#include "features/FeatureTypes.h"
#include "features/Subset.h"

namespace shogun
{
	class CFile;
	class CPreprocessor;
	class CKernel;
	enum EFeatureType;
	enum EFeatureClass;
	enum EFeatureProperty;
}

namespace shogun
{

/** @brief The class Features is the base class of all feature objects.
 *
 * It can be understood as a dense real valued feature matrix (with e.g.
 * columns as single feature vectors), a set of strings, graphs or any other
 * arbitrary collection of objects. As a result this class is kept very general
 * and implements only very weak interfaces to
 *
 * - duplicate the Feature object
 * - obtain the feature type (like F_DREAL, F_SHORT ...)
 * - obtain the feature class (like Simple dense matrices, sparse or strings)
 * - obtain the number of feature "vectors"
 *
 *   In addition it provides helpers to check e.g. for compability of feature objects.
 *
 *   Currently there are 3 general feature classes, which are CSimpleFeatures
 *   (dense matrices), CSparseFeatures (sparse matrices), CStringFeatures (a
 *   set of strings) from which all the specific features like CSimpleFeatures<float64_t>
 *   (dense real valued feature matrices) are derived.
 *
 *   Inherits from Subset to provide feature subset functionality.
 */
class CFeatures : public CSGObject
{
	void init(void);

	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		CFeatures(int32_t size=0);

		/** copy constructor */
		CFeatures(const CFeatures& orig);

		/** constructor
		 *
		 * @param loader File object via which data shall be loaded
		 */
		CFeatures(CFile* loader);

		/** duplicate feature object
		 *
		 * abstract base method
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const=0;

		virtual ~CFeatures();

		/** get feature type
		 *
		 * abstract base method
		 *
		 * @return templated feature type
		 */
		virtual EFeatureType get_feature_type()=0;

		/** get feature class
		 *
		 * abstract base method
		 *
		 * @return feature class like STRING, SIMPLE, SPARSE...
		 */
		virtual EFeatureClass get_feature_class()=0;

		/** add preprocessor
		 *
		 * @param p preprocessor to set
		 * @return something inty
		 */
		virtual int32_t add_preprocessor(CPreprocessor* p);

		/** delete preprocessor from list
		 * caller has to clean up returned preproc
		 *
		 * @param num index of preprocessor in list
		 */
		virtual CPreprocessor* del_preprocessor(int32_t num);

		/** get specified preprocessor
		 *
		 * @param num index of preprocessor in list
		 */
		CPreprocessor* get_preprocessor(int32_t num);

		/** set applied flag for preprocessor
		 *
		 * @param num index of preprocessor in list
		 */
		inline void set_preprocessed(int32_t num) { preprocessed[num]=true; }

		/** get whether specified preprocessor was already applied
		 *
		 * @param num index of preprocessor in list
		 */
		inline bool is_preprocessed(int32_t num) { return preprocessed[num]; }

		/** get the number of applied preprocs
		 *
		 * @return number of applied preprocessors
		 */
		int32_t get_num_preprocessed();

		/** get number of preprocessors
		 *
		 * @return number of preprocessors
		 */
		inline int32_t get_num_preprocessors() { return num_preproc; }

		/** clears all preprocs */
		void clean_preprocessors();

		/** get cache size
		 *
		 * @return cache size
		 */
		inline int32_t get_cache_size() { return cache_size; };

		/** get number of examples/vectors, possibly corresponding to the current subset
		 *
		 * abstract base method
		 *
		 * @return number of examples/vectors (possibly of subset, if implemented)
		 */
		virtual int32_t get_num_vectors()=0;

		/** in case there is a feature matrix allow for reshaping
		 *
		 * NOT IMPLEMENTED!
		 *
		 * @param num_features new number of features
		 * @param num_vectors new number of vectors
		 * @return if reshaping was successful
		 */
		virtual bool reshape(int32_t num_features, int32_t num_vectors) { return false; }

		/** get memory footprint of one feature
		 *
		 * abstract base method
		 *
		 * @return memory footprint of one feature
		 */
		virtual int32_t get_size()=0;

		/** list feature object */
		void list_feature_obj();

		/** load features from file
		 *
		 * @param loader File object via which data shall be loaded
		 */
		virtual void load(CFile* loader)
		{
			SG_SET_LOCALE_C;
			SG_NOTIMPLEMENTED;
			SG_RESET_LOCALE;
		}

		/** save features to file
		 *
		 * @param writer File object via which data shall be saved
		 */
		virtual void save(CFile* writer)
		{
			SG_SET_LOCALE_C;
			SG_NOTIMPLEMENTED;
			SG_RESET_LOCALE;
		}

		/** check feature compatibility
		 *
		 * @param f features to check for compatibility
		 * @return if features are compatible
		 */
		bool check_feature_compatibility(CFeatures* f);

		/** check if features have given property
		 *
		 * @param p feature property
		 * @return if features have given property
		 */
		inline bool has_property(EFeatureProperty p) { return (properties & p) != 0; }

		/** set property
		 *
		 * @param p kernel property to set
		 */
		inline void set_property(EFeatureProperty p)
		{
			properties |= p;
		}

		/** unset property
		 *
		 * @param p kernel property to unset
		 */
		inline void unset_property(EFeatureProperty p)
		{
			properties &= (properties | p) ^ p;
		}

		/*********************************
		 * wrapper for Subset methods
		 * (to avoid mutliple inheritance)
		 ********************************/
		virtual void remove_subset()
		{
			m_subset->remove_subset();
		}

		virtual SGVector<index_t> get_subset() const
		{
			return m_subset->get_subset();
		}

		bool has_subset() const
		{
			return m_subset->has_subset();
		}

		virtual void set_subset(SGVector<index_t> subset)
		{
			m_subset->set_subset(subset);
		}

		inline index_t subset_idx_conversion(index_t idx) const
		{
			return m_subset->subset_idx_conversion(idx);
		}
		/***********************************
		 * End of wrapper for Subset methods
		 **********************************/

	private:
		/** feature properties */
		uint64_t  properties;

		/// size of cache in MB
		int32_t cache_size;

		/// list of preprocessors
		CPreprocessor** preproc;

		/// number of preprocs in list
		int32_t num_preproc;

		/// i'th entry is true if features were already preprocessed with preproc i
		bool* preprocessed;

		/* subset class to enable subset support for this class */
		CSubset* m_subset;
};
}
#endif
