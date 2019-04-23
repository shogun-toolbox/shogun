/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Viktor Gal, Soumyajit De, 
 *          Abhijeet Kislay, Evan Shelhamer, Yuyu Zhang
 */

#ifndef PREPROCESSOR_H_
#define PREPROCESSOR_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/features/FeatureTypes.h>
#include <shogun/features/Features.h>
#include <shogun/lib/common.h>
#include <shogun/transformer/Transformer.h>

namespace shogun
{

class Features;

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
	P_RESCALEFEATURES = 200,
	P_FISHERLDA = 210
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
 * DensePreprocessor. Although, for providing a generic interface for this,
 * an abstract transform() method is there, which sub-classes may choose to use
 * as
 * a wrapper to more specific methods.
 */
class Preprocessor : public Transformer
{
public:
	/** constructor */
	Preprocessor() : Transformer(){};

	/** destructor */
	virtual ~Preprocessor()
	{
	}

	virtual std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace) = 0;

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
