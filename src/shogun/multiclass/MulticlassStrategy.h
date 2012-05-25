/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef MULTICLASSSTRATEGY_H__
#define MULTICLASSSTRATEGY_H__

#include <shogun/base/SGObject.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/RejectionStrategy.h>

namespace shogun
{

class CMulticlassStrategy: public CSGObject
{
public:
	/** constructor */
	CMulticlassStrategy();

	/** destructor */
	virtual ~CMulticlassStrategy() {}

	/** get name */
	virtual const char* get_name() const
	{
		return "MulticlassStrategy";
	};

	/** set number of classes */
	void set_num_classes(int32_t num_classes)
	{
		m_num_classes = num_classes;
	}

	/** get number of classes */
	int32_t get_num_classes() const
	{
		return m_num_classes;
	}

	/** get rejection strategy */
	CRejectionStrategy *get_rejection_strategy()
	{
		SG_REF(m_rejection_strategy);
		return m_rejection_strategy;
	}

	/** set rejection strategy */
	void set_rejection_strategy(CRejectionStrategy *rejection_strategy)
	{
		SG_REF(rejection_strategy);
		SG_UNREF(m_rejection_strategy);
		m_rejection_strategy = rejection_strategy;
	}

	/** start training */
	virtual void train_start(CMulticlassLabels *orig_labels, CBinaryLabels *train_labels);

	/** has more training phase */
	virtual bool train_has_more()=0;

	/** prepare for the next training phase.
	 * @return The subset that should be applied. Return NULL when no subset is needed.
	 */
	virtual SGVector<int32_t> train_prepare_next();

	/** finish training, release resources */
	virtual void train_stop();

	/** decide the final label.
	 * @param outputs a vector of output from each machine (in that order)
	 */
	virtual int32_t decide_label(SGVector<float64_t> outputs)=0;

	/** get number of machines used in this strategy.
	 */
	virtual int32_t get_num_machines()=0;

protected:

	CRejectionStrategy* m_rejection_strategy; ///< rejection strategy
	CBinaryLabels *m_train_labels;    ///< labels used to train the submachines
	CMulticlassLabels *m_orig_labels; ///< original multiclass labels
	int32_t m_train_iter;             ///< index of current iterations
    int32_t m_num_classes;            ///< number of classes in this problem
};

} // namespace shogun

#endif /* end of include guard: MULTICLASSSTRATEGY_H__ */

