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

#ifndef _CPREPROC__H__
#define _CPREPROC__H__

#include "lib/common.h"
#include "base/SGObject.h"
#include "features/Features.h"

class CPreProc;
class CFeatures;

/** class PreProc */
class CPreProc : public CSGObject
{
public:
	/** constructor
	 *
	 * @param name preprocessor's name
	 * @param id preprocessor's id
	 */
	CPreProc(const CHAR* name, const CHAR* id);
	virtual ~CPreProc();

	/// initialize preprocessor from features
	virtual bool init(CFeatures* f)=0;
	
	/// initialize preprocessor from file
	virtual bool load_init_data(FILE* src)=0;

	/// save init-data (like transforamtion matrices etc) to file
	virtual bool save_init_data(FILE* dst)=0;

	/// cleanup
	virtual void cleanup()=0;
	
	/** return feature type with which objects derived 
	from CPreProc can deal */
	virtual EFeatureType get_feature_type()=0;

	/** return feature class
	    like Sparse,Simple,...
	*/
	virtual EFeatureClass get_feature_class()=0;

	/// return the name of the preprocessor
	const CHAR * get_name() const { return preproc_name; }
	
	/// return a FOUR letter id of the preprocessor
	const CHAR * get_id() const { return preproc_id; }

 protected:
 	/** preprocessor name */
	const CHAR* preproc_name;
	/** preprocessor id */
	const CHAR* preproc_id;
};
#endif
