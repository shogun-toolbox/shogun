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

#ifndef MULTICLASSSTRATEGY_H__
#define MULTICLASSSTRATEGY_H__

#include <base/SGObject.h>
#include <labels/BinaryLabels.h>
#include <labels/MulticlassLabels.h>
#include <multiclass/RejectionStrategy.h>
#include <mathematics/Statistics.h>

namespace shogun
{

/** multiclass prob output heuristics in [1]
 * OVA_NORM: simple normalization of probabilites, eq.(6)
 * OVA_SOFTMAX: normalizing using softmax function, eq.(7)
 * OVO_PRICE: proposed by Price et al. see method 1 in [1]
 * OVO_HASTIE: proposed by Hastie et al. see method 2 [9] in [1]
 * OVO_HAMAMURA: proposed by Hamamura et al. see eq.(14) in [1]
 *
 * [1] J. Milgram, M. Cheriet, R.Sabourin, "One Against One" or "One Against One":
 * Which One is Better for Handwriting Recognition with SVMs?
 */
enum EProbHeuristicType
{
	PROB_HEURIS_NONE = 0,
	OVA_NORM = 1,
	OVA_SOFTMAX = 2,
	OVO_PRICE = 3,
	OVO_HASTIE = 4,
	OVO_HAMAMURA = 5
};

/** @brief class MulticlassStrategy used to construct generic
 * multiclass classifiers with ensembles of binary classifiers
 */
class CMulticlassStrategy: public CSGObject
{
public:
	/** constructor */
	CMulticlassStrategy();

	/** constructor
	 * @param prob_heuris probability estimation heuristic
	 */
	CMulticlassStrategy(EProbHeuristicType prob_heuris);

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

	/** decide the final label.
	 * @param outputs a vector of output from each machine (in that order)
	 * @param n_outputs number of outputs
	 */
	virtual SGVector<index_t> decide_label_multiple_output(SGVector<float64_t> outputs, int32_t n_outputs)
	{
		SG_NOTIMPLEMENTED
		return SGVector<index_t>();
	}

	/** get number of machines used in this strategy.
	 */
	virtual int32_t get_num_machines()=0;

	/** get prob output heuristic type */
	EProbHeuristicType get_prob_heuris_type()
	{
		return m_prob_heuris;
	}

	/** set prob output heuristic type
	 * @param prob_heuris type of probability heuristic
	 */
	void set_prob_heuris_type(EProbHeuristicType prob_heuris)
	{
		m_prob_heuris = prob_heuris;
	}

	/** rescale multiclass outputs according to the selected heuristic
	 * NOTE: no matter OVA or OVO, only num_classes rescaled outputs
	 * will be returned as the posteriors
	 * @param outputs a vector of output from each machine (in that order)
	 */
	virtual void rescale_outputs(SGVector<float64_t> outputs)
	{
		SG_NOTIMPLEMENTED
	}

	/** rescale multiclass outputs according to the selected heuristic
	 * this function only being called with OVA_SOFTMAX heuristic
	 * the CStatistics::fit_sigmoid() should be called first
	 * @param outputs a vector of output from each machine (in that order)
	 * @param As fitted sigmoid parameters a one for each machine
	 * @param Bs fitted sigmoid parameters b one for each machine
	 */
	virtual void rescale_outputs(SGVector<float64_t> outputs,
			const SGVector<float64_t> As, const SGVector<float64_t> Bs)
	{
		SG_NOTIMPLEMENTED
	}

private:
	/** initialize variables which will be called by all constructors */
	void init();

protected:

	CRejectionStrategy* m_rejection_strategy; ///< rejection strategy
	CBinaryLabels *m_train_labels;    ///< labels used to train the submachines
	CMulticlassLabels *m_orig_labels; ///< original multiclass labels
	int32_t m_train_iter;             ///< index of current iterations
    int32_t m_num_classes;            ///< number of classes in this problem
	EProbHeuristicType m_prob_heuris; ///< prob output heuristic
};

} // namespace shogun

#endif /* end of include guard: MULTICLASSSTRATEGY_H__ */

