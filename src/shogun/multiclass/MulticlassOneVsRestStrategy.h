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

#ifndef MULTICLASSONEVSRESTSTRATEGY_H__
#define MULTICLASSONEVSRESTSTRATEGY_H__

#include <multiclass/MulticlassStrategy.h>

namespace shogun
{

/** @brief multiclass one vs rest strategy
 * used to train generic multiclass machines
 * for K-class problems with building
 * ensemble of K binary classifiers
 *
 * multiclass probabilistic outputs can be
 * obtained by using the heuristics described in [1]
 *
 * [1] J. Milgram, M. Cheriet, R.Sabourin, "One Against One" or "One Against One":
 * Which One is Better for Handwriting Recognition with SVMs?
 */
class CMulticlassOneVsRestStrategy: public CMulticlassStrategy
{
public:
	/** constructor */
	CMulticlassOneVsRestStrategy();

	/** constructor
	 * @param prob_heuris probability estimation heuristic
	 */
	CMulticlassOneVsRestStrategy(EProbHeuristicType prob_heuris);

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

	/** rescale multiclass outputs according to the selected heuristic
	 * @param outputs a vector of output from each machine (in that order)
	 */
	virtual void rescale_outputs(SGVector<float64_t> outputs);

	/** rescale multiclass outputs according to the selected heuristic
	 * this function only being called with OVA_SOFTMAX heuristic
	 * @param outputs a vector of output from each machine (in that order)
	 * @param As fitted sigmoid parameters a one for each machine
	 * @param Bs fitted sigmoid parameters b one for each machine
	 */
	virtual void rescale_outputs(SGVector<float64_t> outputs,
			const SGVector<float64_t> As, const SGVector<float64_t> Bs);

protected:
	/** OVA normalization heuristic
	 * @param outputs a vector of output from each machine (in that order)
	 */
	void rescale_heuris_norm(SGVector<float64_t> outputs);

	/** OVA softmax heuristic
	 * @param outputs a vector of output from each machine (in that order)
	 * @param As fitted sigmoid parameters a one for each machine
	 * @param Bs fitted sigmoid parameters b one for each machine
	 */
	void rescale_heuris_softmax(SGVector<float64_t> outputs,
			const SGVector<float64_t> As, const SGVector<float64_t> Bs);

};

} // namespace shogun

#endif /* end of include guard: MULTICLASSONEVSRESTSTRATEGY_H__ */

