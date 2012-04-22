#ifndef MULTICLASSSTRATEGY_H__
#define MULTICLASSSTRATEGY_H__

#include <shogun/base/SGObject.h>
#include <shogun/features/Labels.h>
#include <shogun/features/Subset.h>
#include <shogun/features/RejectionStrategy.h>

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
enum EMulticlassStrategy
{
	ONE_VS_REST_STRATEGY,
	ONE_VS_ONE_STRATEGY,
};
#endif

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

	/** get strategy type */
	virtual EMulticlassStrategy get_strategy_type()=0;

	/** start training */
	virtual void train_start(CLabels *orig_labels, CLabels *train_labels)
	{
		if (m_train_labels != NULL)
			SG_ERROR("Stop the previous training task before starting a new one!");
		SG_REF(train_labels);
		m_train_labels=train_labels;
		SG_REF(orig_labels);
		m_orig_labels=orig_labels;
		m_train_iter=0;
	}

	/** has more training phase */
	virtual bool train_has_more()=0;

	/** prepare for the next training phase.
	 * @return The subset that should be applied. Return NULL when no subset is needed.
	 */
	virtual CSubset *train_prepare_next()
	{
		m_train_iter++;
		return NULL;
	}

	/** finish training, release resources */
	virtual void train_stop()
	{
		SG_UNREF(m_train_labels);
		SG_UNREF(m_orig_labels);
	}

	/** decide the final label.
	 * @param outputs a vector of output from each machine (in that order)
	 * @param num_classes number of classes
	 */
	virtual int32_t decide_label(const SGVector<float64_t> &outputs, int32_t num_classes)=0;

	/** get number of machines used in this strategy.
	 */
	virtual int32_t get_num_machines()=0;

protected:
	CLabels *m_train_labels;
	CLabels *m_orig_labels;
	int32_t m_train_iter;
};

class CMulticlassOneVsRestStrategy: public CMulticlassStrategy
{
public:
	/** constructor */
	CMulticlassOneVsRestStrategy();

	/** constructor with rejection strategy */
	CMulticlassOneVsRestStrategy(CRejectionStrategy *rejection_strategy);

	/** destructor */
	virtual ~CMulticlassOneVsRestStrategy() {}

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
	virtual void train_start(CLabels *orig_labels, CLabels *train_labels)
	{
		CMulticlassStrategy::train_start(orig_labels, train_labels);
		m_num_machines=m_orig_labels->get_num_classes();
	}

	/** has more training phase */
	virtual bool train_has_more()
	{
		return m_train_iter < m_num_machines;
	}

	/** prepare for the next training phase.
	 * @return NULL, since no subset is needed in one-vs-rest strategy
	 */ 
	virtual CSubset *train_prepare_next();

	/** decide the final label.
	 * @param outputs a vector of output from each machine (in that order)
	 * @param num_classes number of classes
	 */
	virtual int32_t decide_label(const SGVector<float64_t> &outputs, int32_t num_classes);

	/** get number of machines used in this strategy.
	 * one-vs-rest strategy use one machine for each of the classes.
	 */
	virtual int32_t get_num_machines()
	{
		return m_num_machines;
	}

	/** get strategy type */
	virtual EMulticlassStrategy get_strategy_type()
	{
		return ONE_VS_REST_STRATEGY;
	}

	/** get name */
	virtual const char* get_name() const
	{
		return "MulticlassOneVsRestStrategy";
	};

protected:
	int32_t m_num_machines;
	CRejectionStrategy *m_rejection_strategy;
};

class CMulticlassOneVsOneStrategy: public CMulticlassStrategy
{
public:
	/** constructor */
	CMulticlassOneVsOneStrategy();

	/** destructor */
	virtual ~CMulticlassOneVsOneStrategy() {}

	/** start training */
	virtual void train_start(CLabels *orig_labels, CLabels *train_labels)
	{
		CMulticlassStrategy::train_start(orig_labels, train_labels);
		m_num_classes = m_orig_labels->get_num_classes();
		m_num_machines=m_num_classes*(m_num_classes-1)/2;

		m_train_pair_idx_1 = 0;
		m_train_pair_idx_2 = 1;
	}

	/** has more training phase */
	virtual bool train_has_more()
	{
		return m_train_iter < m_num_machines;
	}

	/** prepare for the next training phase.
	 * @return the subset that should be applied before training.
	 */
	virtual CSubset *train_prepare_next();

	/** decide the final label.
	 * @param outputs a vector of output from each machine (in that order)
	 * @param num_classes number of classes
	 */
	virtual int32_t decide_label(const SGVector<float64_t> &outputs, int32_t num_classes);

	/** get number of machines used in this strategy.
	 * one-vs-one strategy use one machine for each pair of classes.
	 */
	virtual int32_t get_num_machines()
	{
		return m_num_machines;
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

#endif /* end of include guard: MULTICLASSSTRATEGY_H__ */

