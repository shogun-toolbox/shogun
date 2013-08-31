/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef MULTITASKROCEVALUATION_H_
#define MULTITASKROCEVALUATION_H_

#include <shogun/transfer/multitask/TaskRelation.h>
#include <shogun/evaluation/ROCEvaluation.h>

namespace shogun
{

/** @brief Class MultitaskROCEvalution used to evaluate ROC
 * (Receiver Operating Characteristic) and an area
 * under ROC curve (auROC) of each task separately.
 *
 */
class CMultitaskROCEvaluation: public CROCEvaluation
{
public:
	/** constructor */
	CMultitaskROCEvaluation() :
		CROCEvaluation(), m_task_relation(NULL), m_tasks_indices(NULL),
		m_num_tasks(0)
	{
	}

	/** constructor */
	CMultitaskROCEvaluation(CTaskRelation* task_relation) :
		CROCEvaluation(), m_task_relation(NULL), m_tasks_indices(NULL),
		m_num_tasks(0)
	{
		set_task_relation(task_relation);
	}

	/** destructor */
	virtual ~CMultitaskROCEvaluation()
	{
		SG_FREE(m_tasks_indices);
	}

	/** set task relation */
	void set_task_relation(CTaskRelation* task_relation)
	{
		SG_REF(task_relation);
		SG_UNREF(m_task_relation);
		m_task_relation = task_relation;
	}

	/** get task relation */
	CTaskRelation* get_task_relation() const
	{
		SG_REF(m_task_relation);
		return m_task_relation;
	}

	/** set absolute indices of labels to be evaluated next
	 * used by multitask evaluations
	 *
	 * @param indices indices
	 */
	virtual void set_indices(SGVector<index_t> indices);

	/** get name */
	virtual const char* get_name() const { return "MultitaskROCEvaluation"; };

	/** evaluate ROC and auROC
	 * @param predicted labels
	 * @param ground_truth labels assumed to be correct
	 * @return auROC
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth);

	/** get evaluation direction */
	virtual EEvaluationDirection get_evaluation_direction() const
	{
		return ED_MAXIMIZE;
	}

protected:

	/** task relation */
	CTaskRelation* m_task_relation;

	/** indices */
	SGVector<index_t>* m_tasks_indices;

	/** num tasks */
	int32_t m_num_tasks;
};

}

#endif /* MULTITASKROCEVALUATION_H_ */
