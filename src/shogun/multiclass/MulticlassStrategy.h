/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Shell Hu, Sergey Lisitsyn, Soeren Sonnenburg, 
 *          Heiko Strathmann, Yuyu Zhang, Bjoern Esser
 */

#ifndef MULTICLASSSTRATEGY_H__
#define MULTICLASSSTRATEGY_H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/RejectionStrategy.h>
#include <shogun/mathematics/Statistics.h>

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
class MulticlassStrategy: public SGObject
{
public:
	/** constructor */
	MulticlassStrategy();

	/** constructor
	 * @param prob_heuris probability estimation heuristic
	 */
	MulticlassStrategy(EProbHeuristicType prob_heuris);

	/** destructor */
	virtual ~MulticlassStrategy() {}

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
	std::shared_ptr<RejectionStrategy >get_rejection_strategy()
	{
		
		return m_rejection_strategy;
	}

	/** set rejection strategy */
	void set_rejection_strategy(std::shared_ptr<RejectionStrategy >rejection_strategy)
	{
		
		
		m_rejection_strategy = rejection_strategy;
	}

	/** start training */
	virtual void train_start(std::shared_ptr<MulticlassLabels >orig_labels, std::shared_ptr<BinaryLabels >train_labels);

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
	 * the Statistics::fit_sigmoid() should be called first
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

	std::shared_ptr<RejectionStrategy> m_rejection_strategy; ///< rejection strategy
	std::shared_ptr<BinaryLabels >m_train_labels;    ///< labels used to train the submachines
	std::shared_ptr<MulticlassLabels >m_orig_labels; ///< original multiclass labels
	int32_t m_train_iter;             ///< index of current iterations
    int32_t m_num_classes;            ///< number of classes in this problem
	EProbHeuristicType m_prob_heuris; ///< prob output heuristic
};

} // namespace shogun

#endif /* end of include guard: MULTICLASSSTRATEGY_H__ */

