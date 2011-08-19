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

#ifndef _CPREPROCESSOR__H__
#define _CPREPROCESSOR__H__

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/features/FeatureTypes.h>
#include <shogun/features/Features.h>

namespace shogun
{

class CFeatures;

enum EFeatureType;

enum EFeatureClass;

/** enumeration of possible preprocessor types */
enum EPreprocessorType
{
	P_UNKNOWN=0,
	P_NORMONE=10,
	P_LOGPLUSONE=20,
	P_SORTWORDSTRING=30,
	P_SORTULONGSTRING=40,
	P_SORTWORD=50,
	P_PRUNEVARSUBMEAN=60,
	P_DECOMPRESSSTRING=70,
	P_DECOMPRESSCHARSTRING=80,
	P_DECOMPRESSBYTESTRING=90,
	P_DECOMPRESSWORDSTRING=100,
	P_DECOMPRESSULONGSTRING=110,
	P_RANDOMFOURIERGAUSS=120,
	P_PCA=130,
	P_KERNELPCA=140,
	P_NORMDERIVATIVELEM3=150,
	P_DIMENSIONREDUCTIONPREPROCESSOR=160,
	P_MULTIDIMENSIONALSCALING=170,
	P_LOCALLYLINEAREMBEDDING=180,
	P_ISOMAP=190,
	P_HESSIANLOCALLYLINEAREMBEDDING=200,
	P_LOCALTANGENTSPACEALIGNMENT=210,
	P_LAPLACIANEIGENMAPS=220
};

class CFeatures;

/** @brief Class Preprocessor defines a preprocessor interface.
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
 * CSimplePreprocessor .
 */
class CPreprocessor : public CSGObject
{
public:
	/** constructor
	 */
	CPreprocessor();

	/** destructor
	 */
	virtual ~CPreprocessor();

	/// initialize preprocessor from features
	virtual bool init(CFeatures* f)=0;

	/// cleanup
	virtual void cleanup()=0;
	
	/** return feature type with which objects derived 
	from CPreprocessor can deal */
	virtual EFeatureType get_feature_type()=0;

	/** return feature class
	    like Sparse,Simple,...
	*/
	virtual EFeatureClass get_feature_class()=0;

	/// return a type of preprocessor
	virtual EPreprocessorType get_type() const=0;
};
}
#endif // _CPREPROCESSOR__H__
