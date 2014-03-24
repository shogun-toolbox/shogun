/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Written (W) 2013 Shell Hu and Heiko Strathmann
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/multiclass/MulticlassStrategy.h>
#include <shogun/lib/config.h>

namespace shogun
{

/** @brief multiclass one vs one strategy
 * used to train generic multiclass machines
 * for K-class problems with building
 * voting-based ensemble of K*(K-1)
 * binary classifiers
 * multiclass probabilistic outputs can be
 * obtained by using the heuristics described in [1]
 *
 * [1] J. Milgram, M. Cheriet, R.Sabourin, "One Against One" or "One Against One":
 * Which One is Better for Handwriting Recognition with SVMs?
 */
class CMulticlassOneVsOneStrategy: public CMulticlassStrategy
{
public:
	/** constructor */
	CMulticlassOneVsOneStrategy();

	/** constructor
	 * @param prob_heuris probability estimation heuristic
	 */
	CMulticlassOneVsOneStrategy(EProbHeuristicType prob_heuris);

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

	/** rescale multiclass outputs according to the selected heuristic
	 * @param outputs a vector of output from each machine (in that order)
	 * which will be resized to length of num_classes if heuristic is set
	 */
	virtual void rescale_outputs(SGVector<float64_t> outputs);

	/** set the number of classes, since the number of machines totally
	 * depends on the number of classes, which will also be set.
	 * @param num_classes number of classes
	 */
	void set_num_classes(int32_t num_classes)
	{
		CMulticlassStrategy::set_num_classes(num_classes);
		m_num_machines = m_num_classes*(m_num_classes-1)/2;
	}

protected:
	/** OVO Price's heuristic see [1]
	 * @param outputs a vector of output from each machine (in that order)
	 * @param indx1 indices of 1st involved class of training machines
	 * @param indx2 indices of 2nd involved class of training machines
	 */
	void rescale_heuris_price(SGVector<float64_t> outputs,
		const SGVector<int32_t> indx1, const SGVector<int32_t> indx2);

	/** OVO Hastie's heuristic see [1]
	 * @param outputs a vector of output from each machine (in that order)
	 * @param indx1 indices of 1st involved class of training machines
	 * @param indx2 indices of 2nd involved class of training machines
	 */
	void rescale_heuris_hastie(SGVector<float64_t> outputs,
		const SGVector<int32_t> indx1, const SGVector<int32_t> indx2);

	/** OVO Hamamura's heuristic see [1]
	 * @param outputs a vector of output from each machine (in that order)
	 * @param indx1 indices of 1st involved class of training machines
	 * @param indx2 indices of 2nd involved class of training machines
	 */
	void rescale_heuris_hamamura(SGVector<float64_t> outputs,
		const SGVector<int32_t> indx1, const SGVector<int32_t> indx2);

private:
	/** register parameters */
	void register_parameters();

protected:
	int32_t m_num_machines;     ///< number of machines
	int32_t m_train_pair_idx_1; ///< 1st index of current submachine being trained
	int32_t m_train_pair_idx_2; ///< 2nd index of current submachine being trained
	SGVector<int32_t> m_num_samples; ///< number of samples per machine
};

} // namespace shogun
