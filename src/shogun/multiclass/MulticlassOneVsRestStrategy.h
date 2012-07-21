/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef MULTICLASSONEVSRESTSTRATEGY_H__
#define MULTICLASSONEVSRESTSTRATEGY_H__

#include <shogun/multiclass/MulticlassStrategy.h>

namespace shogun
{

class CMulticlassOneVsRestStrategy: public CMulticlassStrategy
{
public:
	/** constructor */
	CMulticlassOneVsRestStrategy();

	/** destructor */
	virtual ~CMulticlassOneVsRestStrategy() {}

	/** start training */
	virtual void train_start(CMulticlassLabels *orig_labels, CBinaryLabels *train_labels)
	{
		CMulticlassStrategy::train_start(orig_labels, train_labels);
	}

	/** has more training phase */
	virtual bool train_has_more()
	{
		return m_train_iter < m_num_classes;
	}

	/** prepare for the next training phase.
	 * @return NULL, since no subset is needed in one-vs-rest strategy
	 */
	virtual SGVector<int32_t> train_prepare_next();

	/** decide the final label.
	 * @param outputs a vector of output from each machine (in that order)
	 */
	virtual int32_t decide_label(SGVector<float64_t> outputs);
	
	/** decide the final label.
	 * @param outputs a vector of output from each machine (in that order)
	 * @param n_outputs number of outputs
	 */
	virtual SGVector<index_t> decide_label_multiple_output(SGVector<float64_t> outputs, int32_t n_outputs);

	/** get number of machines used in this strategy.
	 */
	virtual int32_t get_num_machines()
	{
		return m_num_classes;
	}

	/** get name */
	virtual const char* get_name() const
	{
		return "MulticlassOneVsRestStrategy";
	};

};

} // namespace shogun

#endif /* end of include guard: MULTICLASSONEVSRESTSTRATEGY_H__ */

