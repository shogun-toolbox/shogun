/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */


#ifndef _CFEATURES__H__
#define _CFEATURES__H__

#include "lib/common.h"
#include "base/SGObject.h"

#include "preproc/PreProc.h"

class CPreProc;
class CFeatures;

/** class Features
 * Features can just be DREALs, SHORT or STRINGs, FILES, or...
 */
class CFeatures : public CSGObject
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		CFeatures(INT size);

		/** copy constructor */
		CFeatures(const CFeatures& orig);

		/** constructor
		 *
		 * @param fname filename to load features from
		 */
		CFeatures(CHAR* fname);

		/** duplicate feature object
		 *
		 * abstract base method
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const=0 ;

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
		virtual INT add_preproc(CPreProc* p);

		/** delete preprocessor from list
		 * caller has to clean up returned preproc
		 *
		 * @param num index of preprocessor in list
		 */
		virtual CPreProc* del_preproc(INT num);

		/** get specified preprocessor
		 *
		 * @param num index of preprocessor in list
		 */
		CPreProc* get_preproc(INT num);

		/** set applied flag for preprocessor
		 *
		 * @param num index of preprocessor in list
		 */
		inline void set_preprocessed(INT num) { preprocessed[num]=true; }

		/** get whether specified preprocessor was already applied
		 *
		 * @param num index of preprocessor in list
		 */
		inline bool is_preprocessed(INT num) { return preprocessed[num]; }

		/** get the number of applied preprocs
		 *
		 * @return number of applied preprocessors
		 */
		INT get_num_preprocessed();

		/** get number of preprocessors
		 *
		 * @return number of preprocessors
		 */
		inline INT get_num_preproc() { return num_preproc; }

		/** clears all preprocs */
		void clean_preprocs();

		/** get cache size
		 *
		 * @return cache size
		 */
		inline INT get_cache_size() { return cache_size; };

		/** get number of examples/vectors
		 *
		 * abstract base method
		 *
		 * @return number of examples/vectors
		 */
		virtual INT get_num_vectors()=0 ;

		/** in case there is a feature matrix allow for reshaping
		 *
		 * NOT IMPLEMENTED!
		 *
		 * @param num_features new number of features
		 * @param num_vectors new number of vectors
		 * @return if reshaping was succesful
		 */
		virtual bool reshape(INT num_features, INT num_vectors) { return false; }

		/** get memory footprint of one feature
		 *
		 * abstrace base method
		 *
		 * @return memory footprint of one feature
		 */
		virtual INT get_size()=0;

		/** list feature object */
		void list_feature_obj();

		/** load features from file
		 *
		 * @param fname filename to load from
		 * @return if loading was successful
		 */
		virtual bool load(CHAR* fname);

		/** save features to file
		 *
		 * @param fname filename to save to
		 * @return if saving was successful
		 */
		virtual bool save(CHAR* fname);

		/** check feature compatibility
		 *
		 * @param f features to check for compatibility
		 * @return if features are compatible
		 */
		bool check_feature_compatibility(CFeatures* f);

	private:
		/// size of cache in MB
		INT cache_size;

		/// list of preprocessors
		CPreProc** preproc;

		/// number of preprocs in list
		INT num_preproc;

		/// i'th entry is true if features were already preprocessed with preproc i
		bool* preprocessed;
};
#endif
