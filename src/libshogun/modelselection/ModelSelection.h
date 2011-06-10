/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __MODELSELECTION_H_
#define __MODELSELECTION_H_

#include "base/SGObject.h"

namespace shogun
{

class CSplittingStrategy;
class CEvaluation;
class CModelSelectionParameters;
class CParameterCombination;
class CMachine;

/** enum which is used to define whether an evaluation criterium has to be
 * minimised or maximised */
enum EEvaluationDirection
{
	MS_UNDEFINED, MS_MINIMIZE, MS_MAXIMIZE
};

class CModelSelection: public CSGObject
{
public:
	CModelSelection();
	CModelSelection(CMachine* machine, CEvaluation* evaluation_criterium,
			EEvaluationDirection eval_direction,
			CModelSelectionParameters* model_parameters,
			CSplittingStrategy* splitting_strategy);
	virtual ~CModelSelection();

	virtual CParameterCombination* select_model();

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 * @return name of the SGSerializable
	 */
	inline virtual const char* get_name() const
	{
		return "ModelSelection";
	}

protected:
	CMachine* m_machine;
	CEvaluation* m_evaluation_criterium;
	EEvaluationDirection m_evaluation_direction;
	CModelSelectionParameters* m_model_parameters;
	CSplittingStrategy* m_splitting_strategy;
};

}

#endif /* __MODELSELECTION_H_ */
