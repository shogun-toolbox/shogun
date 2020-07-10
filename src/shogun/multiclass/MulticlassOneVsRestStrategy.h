/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Shell Hu, Sergey Lisitsyn, Yuyu Zhang, Bjoern Esser, 
 *          Soeren Sonnenburg
 */

#ifndef MULTICLASSONEVSRESTSTRATEGY_H__
#define MULTICLASSONEVSRESTSTRATEGY_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/MulticlassStrategy.h>

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
class MulticlassOneVsRestStrategy: public MulticlassStrategy
{
public:
	/** constructor */
	MulticlassOneVsRestStrategy();

	/** constructor
	 * @param prob_heuris probability estimation heuristic
	 */
	MulticlassOneVsRestStrategy(EProbHeuristicType prob_heuris);

	/** destructor */
	~MulticlassOneVsRestStrategy() override {}

	/** start training */
	void train_start(std::shared_ptr<MulticlassLabels >orig_labels, std::shared_ptr<BinaryLabels >train_labels) override
	{
		MulticlassStrategy::train_start(orig_labels, train_labels);
	}

	/** has more training phase */
	bool train_has_more() override
	{
		return m_train_iter < m_num_classes;
	}

	/** prepare for the next training phase.
	 * @return NULL, since no subset is needed in one-vs-rest strategy
	 */
	SGVector<int32_t> train_prepare_next() override;

	/** decide the final label.
	 * @param outputs a vector of output from each machine (in that order)
	 */
	int32_t decide_label(SGVector<float64_t> outputs) override;

	/** decide the final label.
	 * @param outputs a vector of output from each machine (in that order)
	 * @param n_outputs number of outputs
	 */
	SGVector<index_t> decide_label_multiple_output(SGVector<float64_t> outputs, int32_t n_outputs) override;

	/** get number of machines used in this strategy.
	 */
	int32_t get_num_machines() override
	{
		return m_num_classes;
	}

	/** get name */
	const char* get_name() const override
	{
		return "MulticlassOneVsRestStrategy";
	};

	/** rescale multiclass outputs according to the selected heuristic
	 * @param outputs a vector of output from each machine (in that order)
	 */
	void rescale_outputs(SGVector<float64_t> outputs) override;

	/** rescale multiclass outputs according to the selected heuristic
	 * this function only being called with OVA_SOFTMAX heuristic
	 * @param outputs a vector of output from each machine (in that order)
	 * @param As fitted sigmoid parameters a one for each machine
	 * @param Bs fitted sigmoid parameters b one for each machine
	 */
	void rescale_outputs(SGVector<float64_t> outputs,
			const SGVector<float64_t> As, const SGVector<float64_t> Bs) override;

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
			const SGVector<float64_t>& As, const SGVector<float64_t>& Bs);

};

} // namespace shogun

#endif /* end of include guard: MULTICLASSONEVSRESTSTRATEGY_H__ */

