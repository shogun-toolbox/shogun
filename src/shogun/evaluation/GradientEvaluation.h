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

namespace shogun {

class CGradientEvaluation: public shogun::CMachineEvaluation {
public:
	CGradientEvaluation();
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

private:
	CDifferentiableFunction* diff;
};

} /* namespace shogun */
#endif /* GRADIENTEVALUATION_H_ */
