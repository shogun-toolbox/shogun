/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/multiclass/MulticlassStrategy.h>

namespace shogun
{
	
class CMulticlassOneVsOneStrategy: public CMulticlassStrategy
{
public:
	/** constructor */
	CMulticlassOneVsOneStrategy();

	/** destructor */
	virtual ~CMulticlassOneVsOneStrategy() {}

	/** start training */
	virtual void train_start(CLabels *orig_labels, CLabels *train_labels);

	/** has more training phase */
	virtual bool train_has_more();

	/** prepare for the next training phase.
	 * @return the subset that should be applied before training.
	 */
	virtual SGVector<int32_t> train_prepare_next();

	/** decide the final label.
	 * @param outputs a vector of output from each machine (in that order)
	 * @param num_classes number of classes
	 */
	virtual int32_t decide_label(const SGVector<float64_t> &outputs, int32_t num_classes);

	/** get number of machines used in this strategy.
	 * @param num_classes number of classes in this problem
	 */
	virtual int32_t get_num_machines(int32_t num_classes)
	{
		return num_classes*(num_classes-1)/2;
	}

	/** get strategy type */
	virtual EMulticlassStrategy get_strategy_type()
	{
		return ONE_VS_ONE_STRATEGY;
	}

	/** get name */
	virtual const char* get_name() const
	{
		return "MulticlassOneVsOneStrategy";
	};

protected:
	int32_t m_num_machines;
	int32_t m_num_classes;
	int32_t m_train_pair_idx_1;
	int32_t m_train_pair_idx_2;
};

} // namespace shogun
