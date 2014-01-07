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

#ifndef __GUIPREPROC_H__
#define __GUIPREPROC_H__

#include <lib/config.h>
#include <lib/List.h>
#include <base/SGObject.h>
#include <preprocessor/Preprocessor.h>

namespace shogun
{
class CSGInterface;

/** @brief UI preprocessor */
class CGUIPreprocessor : public CSGObject
{
	public:
		/** constructor */
		CGUIPreprocessor() { };
		/** constructor
		 * @param interface
		 */
		CGUIPreprocessor(CSGInterface* interface);
		/** destructor */
		~CGUIPreprocessor();

		/** create generic Preprocessor */
		CPreprocessor* create_generic(EPreprocessorType type);
		/** create preproc PruneVarSubMean */
		CPreprocessor* create_prunevarsubmean(bool divide_by_std=false);
		/** create preproc PCA */
		CPreprocessor* create_pca(bool do_whitening, float64_t threshold);

		/** add new preproc to list */
		bool add_preproc(CPreprocessor* preproc);
		/** delete last preproc in list */
		bool del_preproc();
		/** clean all preprocs from list */
		bool clean_preproc();

		/** attach preprocessor to TRAIN/TEST feature obj.
		 *  it will also preprocess train/test data
		 *  when a feature matrix is available
		 */
		bool attach_preproc(char* target, bool do_force=false);

		/** @return object name */
		virtual const char* get_name() const { return "GUIPreprocessor"; }

	protected:
		/** preprocess features
		 * @param trainfeat
		 * @param testfeat
		 * @param force
		 */
		bool preprocess_features(CFeatures* trainfeat, CFeatures* testfeat, bool force);
		/** preproc all features
		 * @param f
		 * @param force
		 */
		bool preproc_all_features(CFeatures* f, bool force);

		/** preprocs */
		CList* preprocs;
		/** ui */
		CSGInterface* ui;
};
}
#endif
