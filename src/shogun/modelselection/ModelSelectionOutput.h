/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 */

#ifndef __MODELSELECTIONOUTPUT_H_
#define __MODELSELECTIONOUTPUT_H_

#include <shogun/base/SGObject.h>
#include <shogun/machine/Machine.h>
#include <shogun/lib/SGVector.h>
#include <shogun/labels/Labels.h>
#include <shogun/evaluation/Evaluation.h>

namespace shogun
{

/**
 * @brief */
class CModelSelectionOutput: public CSGObject
{
public:

	/** constructor */
	CModelSelectionOutput();

	/** destructor */
	virtual ~CModelSelectionOutput();

	/** get name */
	virtual const char* get_name() const { return "ModelSelectionOutput"; }

	/** output train indices */
	void output_train_indices(SGVector<index_t> indices);
	/** output test indices */
	void output_test_indices(SGVector<index_t> indices);
	/** output trained machine */
	void output_trained_machine(CMachine* machine);
	/** output test result */
	void output_test_result(CLabels* results);
	/** output test true result */
	void output_test_true_result(CLabels* results);
	/** output evaluate result */
	void output_evaluate_result(float64_t result);

	/** add custom evaluation */
	void add_custom_evaluation(CEvaluation* evaluation);
	/** output custom evaluations */
	void output_custom_evaluations(CLabels* results, CLabels* truth);

protected:

	/** custom evaluations */
	CDynamicObjectArray* m_custom_evaluations;

};

}

#endif /* __MODELSELECTIONOUTPUT_H_ */
