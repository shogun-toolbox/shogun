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
	virtual void train_start(CMulticlassLabels *orig_labels, CBinaryLabels *train_labels);

	/** has more training phase */
	virtual bool train_has_more();

	/** prepare for the next training phase.
	 * @return the subset that should be applied before training.
	 */
	virtual SGVector<int32_t> train_prepare_next();

	/** decide the final label.
	 * @param outputs a vector of output from each machine (in that order)
	 */
	virtual int32_t decide_label(SGVector<float64_t> outputs);

	/** get number of machines used in this strategy.
	 */
	virtual int32_t get_num_machines()
	{
		return m_num_classes*(m_num_classes-1)/2;
	}

	/** get name */
	virtual const char* get_name() const
	{
		return "MulticlassOneVsOneStrategy";
	};

protected:
	int32_t m_num_machines;     ///< number of machines
	int32_t m_train_pair_idx_1; ///< 1st index of current submachine being trained
	int32_t m_train_pair_idx_2; ///< 2nd index of current submachine being trained
};

} // namespace shogun
