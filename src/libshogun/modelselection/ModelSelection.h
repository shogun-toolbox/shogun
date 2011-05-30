/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __CMODELSELECTION_H_
#define __CMODELSELECTION_H_

#include "base/SGObject.h"
#include "features/Features.h"
#include "features/Labels.h"
#include "machine/Machine.h"
#include "evaluation/Evaluation.h"
#include "modelselection/ModelSelectionParameters.h"

namespace shogun
{

class CModelSelection: public CSGObject
{
public:
	CModelSelection();

	CModelSelection(CMachine* machine, CFeatures* features,
			CLabels* labels);

	~CModelSelection();

	void set_evaluation_criteria(CEvaluation* criteria);
	void set_splitting_strategy();

	void set_parameters(CModelSelectionParameters* param_tree_root);

	void run();

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 * @return name of the SGSerializable
	 */
	inline virtual const char* get_name() const
	{
		return "ModelSelection";
	}

private:
	CMachine* m_machine;
	CFeatures* m_features;
	CLabels* m_labels;

	CModelSelectionParameters* m_param_tree_root;
};
}

#endif  /* __CMODELSELECTION_H_ */
