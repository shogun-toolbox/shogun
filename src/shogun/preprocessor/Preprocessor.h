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

#ifndef PREPROCESSOR_H_
#define PREPROCESSOR_H_

#include <lib/common.h>
#include <base/SGObject.h>
#include <features/FeatureTypes.h>
#include <features/Features.h>

namespace shogun
{

class CFeatures;

/** enumeration of possible preprocessor types
 * used by Shogun UI
 *
 * Note to developer: any new preprocessor should be added here.
 */
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
	P_SUMONE=170,
	P_HOMOGENEOUSKERNELMAP = 180,
	P_PNORM = 190,
	P_RESCALEFEATURES = 200
};

/** @brief Class Preprocessor defines a preprocessor interface.
 *
 * Preprocessors are transformation functions that doesn't change the domain of
 * the input features.  These functions can be applied in-place if the input
 * features fit in memory or can be applied on-the-fly when (depending on
 * features) a feature caching strategy is applied. However, if the individual
 * features are in \f$\bf{R}\f$ they have to stay in \f$\bf{R}\f$ although the
 * dimensionality of the feature vectors is allowed to be changed.
 *
 * As preprocessors might need a certain initialization they may expect that
 * the init() function is called before anything else. The actual preprocessing
 * is feature type dependent and thus coordinated in the sub-classes, cf. e.g.
 * CDensePreprocessor.
 */
class CPreprocessor : public CSGObject
{
public:
	/** constructor */
	CPreprocessor() : CSGObject()
	{
	};

	/** destructor */
	virtual ~CPreprocessor()
	{
	}

	/** initialize preprocessor with features */
	virtual bool init(CFeatures* features)=0;

	/** clean-up. should be called (if necessary) after processing */
	virtual void cleanup()=0;

	/** @return type of objects preprocessor can deal with */
	virtual EFeatureType get_feature_type()=0;

	/** @return class of features preprocessor deals with */
	virtual EFeatureClass get_feature_class()=0;

	/** @return the actual type of the preprocessor */
	virtual EPreprocessorType get_type() const=0;
};
}
#endif // PREPROCESSOR_H_
