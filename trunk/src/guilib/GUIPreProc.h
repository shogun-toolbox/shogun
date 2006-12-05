/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __GUIPREPROC_H__
#define __GUIPREPROC_H__

#include "preproc/PreProc.h"
#include "lib/List.h"

class CGUI;

class CGUIPreProc
{
	public:
		CGUIPreProc(CGUI*);
		~CGUIPreProc();

		bool add_preproc(CHAR* param);
		bool del_preproc(CHAR* param);
		bool clean_preproc(CHAR* param);

		bool load(CHAR* param);
		bool save(CHAR* param);

		/** attach preprocessor to TRAIN/TEST feature obj.
		 *  it will also preprocess train/test data
		 *  when a feature matrix is available
		 */
		bool attach_preproc(CHAR* param);

	protected:
		bool preprocess_features(CFeatures* trainfeat, CFeatures* testfeat, bool force);
		bool preproc_all_features(CFeatures* f, bool force);

		CList<CList<CPreProc*>*>* attached_preprocs_lists;
		CList<CPreProc*>* preprocs;
		CGUI* gui ;
};
#endif
