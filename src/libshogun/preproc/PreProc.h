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

#ifndef _CPREPROC__H__
#define _CPREPROC__H__

#include "lib/common.h"
#include "base/SGObject.h"
#include "features/FeatureTypes.h"
#include "features/Features.h"

namespace shogun
{
	class CFeatures;
	enum EFeatureType;
	enum EFeatureClass;

enum EPreProcType
{
	P_UNKNOWN=0,
	P_NORMONE=10,
	P_LOGPLUSONE=20,
	P_SORTWORDSTRING=30,
	P_SORTULONGSTRING=40,
	P_SORTWORD=50,
	P_PRUNEVARSUBMEAN=60,
	P_DECOMPRESSCHARSTRING=70,
	P_DECOMPRESSBYTESTRING=80,
	P_DECOMPRESSWORDSTRING=90,
	P_DECOMPRESSULONGSTRING=100,
};

class CFeatures;

/** @brief Class PreProc defines a preprocessor interface.
 *
 * Preprocessors are transformation functions that don't change the domain of
 * the input features.  These functions can be applied in-place if the input
 * features fit in memory or can be applied on-the-fly when (depending on
 * features) a feature caching strategy is applied. However, if the individual
 * features are in \f$\bf{R}\f$ they have to stay in \f$\bf{R}\f$ although the
 * dimensionality of the feature vectors is allowed change.
 *
 * As preprocessors might need a certain initialization they may expect that
 * the init() function is called before anything else. The actual preprocessing
 * is feature type dependent and thus coordinated in the sub-classes, cf. e.g.
 * CSimplePreProc .
 */
class CPreProc : public CSGObject
{
public:
	/** constructor
	 *
	 * @param name preprocessor's name
	 * @param id preprocessor's id
	 */
	CPreProc(const char* name, const char* id);
	virtual ~CPreProc();

	/// initialize preprocessor from features
	virtual bool init(CFeatures* f)=0;

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
	virtual const char* get_name() const { return preproc_name; }
	
	/// return a FOUR letter id of the preprocessor
	const char * get_id() const { return preproc_id; }

 protected:
 	/** preprocessor name */
	const char* preproc_name;
	/** preprocessor id */
	const char* preproc_id;
};
}
#endif
