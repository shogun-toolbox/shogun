/*
 * GradientEvaluation.h
 *
 *  Created on: Jun 15, 2012
 *      Author: jacobw
 */

#ifndef GRADIENTEVALUATION_H_
#define GRADIENTEVALUATION_H_

#include "MachineEvaluation.h"
#include <shogun/evaluation/DifferentiableFunction.h>
#include <shogun/evaluation/GradientResult.h>


namespace shogun {

class CGradientEvaluation: public shogun::CMachineEvaluation {
public:
	CGradientEvaluation();

	/** constructor
	 * @param machine learning machine to use
	 * @param features features to use for cross-validation
	 * @param labels labels that correspond to the features
	 * @param splitting_strategy splitting strategy to use
	 * @param evaluation_criterion evaluation criterion to use
	 * @param autolock whether machine should be auto-locked before evaluation
	 */
	CGradientEvaluation(CMachine* machine, CFeatures* features, CLabels* labels,
			CSplittingStrategy* splitting_strategy,
			CEvaluation* evaluation_criterion, bool autolock=true);


	virtual ~CGradientEvaluation();

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
	virtual const char* get_name() const
	{
		return "GradientEvaluation";
	}

	virtual CEvaluationResult* evaluate();

	CDifferentiableFunction* diff;
};

} /* namespace shogun */
#endif /* GRADIENTEVALUATION_H_ */
