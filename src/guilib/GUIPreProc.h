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

#include "lib/config.h"

#ifndef HAVE_SWIG
#include "base/SGObject.h"

#include "preproc/PreProc.h"
#include "lib/List.h"

class CSGInterface;

class CGUIPreProc : public CSGObject
{
	public:
		CGUIPreProc(CSGInterface* interface);
		~CGUIPreProc();

		/** create generic PreProc */
		CPreProc* create_generic(EPreProcType type);
		/** create preproc PruneVarSubMean */
		CPreProc* create_prunevarsubmean(bool divide_by_std);
		/** create preproc PCACUT */
		CPreProc* create_pcacut(bool do_whitening, DREAL threshold);

		/** add new preproc to list */
		bool add_preproc(CPreProc* preproc);
		/** delete last preproc in list */
		bool del_preproc();
		/** clean all preprocs from list */
		bool clean_preproc();

		/** load preproc from file */
		bool load(CHAR* filename);
		/** save preproc to file */
		bool save(CHAR* filename, INT num_preprocs);

		/** attach preprocessor to TRAIN/TEST feature obj.
		 *  it will also preprocess train/test data
		 *  when a feature matrix is available
		 */
		bool attach_preproc(CHAR* target, bool do_force=false);

	protected:
		bool preprocess_features(CFeatures* trainfeat, CFeatures* testfeat, bool force);
		bool preproc_all_features(CFeatures* f, bool force);

		CList<CList<CPreProc*>*>* attached_preprocs_lists;
		CList<CPreProc*>* preprocs;
		CSGInterface* ui;
};
#endif //HAVE_SWIG
#endif
