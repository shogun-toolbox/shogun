/*
 * GradientModelSelection.h
 *
 *  Created on: Jun 15, 2012
 *      Author: jacobw
 */

#ifndef GRADIENTMODELSELECTION_H_
#define GRADIENTMODELSELECTION_H_

#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/modelselection/ModelSelection.h>
#include <shogun/base/DynArray.h>
#include <shogun/evaluation/GradientResult.h>


namespace shogun {

class CGradientModelSelection: public shogun::CModelSelection {
public:
	CGradientModelSelection(CModelSelectionParameters* model_parameters,
			CMachineEvaluation* machine_eval);

	CGradientModelSelection();

	virtual ~CGradientModelSelection();

	virtual CParameterCombination* select_model(bool print_state=false);

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
	virtual const char* get_name()
	{
		return "GradientModelSelection";
	}

private:

typedef struct {
    double a, b;
} nlopt_constraint_data;

double nlopt_const(unsigned n, const double *x, double *grad, void *data);


CParameterCombination* current_combination;

CParameterCombination* best_combination;

CGradientResult best_result;



};

}
/* namespace shogun */
#endif /* GRADIENTMODELSELECTION_H_ */
