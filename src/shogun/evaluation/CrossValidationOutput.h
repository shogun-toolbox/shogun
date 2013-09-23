/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Written (W) 2012 Heiko Strathmann
 *
 */

#ifndef __CROSSVALIDATIONOUTPUT_H_
#define __CROSSVALIDATIONOUTPUT_H_

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

class CMachine;
class CLabels;
class CEvaluation;

/** @brief Class for managing individual folds in cross-validation.
 *
 * It is often desired to print/save informations that occur during individual
 * folds in cross-validation, such as indices, parameters of underlying
 * machine etc. This abstract base class might be called from the
 * CCrossValidation class after each fold in order to collect these things.
 * Different implementations then could output the informations, or even store
 * them to make them accessible later. Since is different for every underlying
 * machine, individual sub-classes have to handle this separately.
 * When writing new subclasses, try to make the design as inheritance based
 * as possible, such that future sub-sub-classes can use yours.
 * Note that subclasses only need to implement methods that they need since all
 * methods in this class are already implemented empty, except for num/fold
 * stuff. These number of runs/folds and their current indices are stored since
 * these might be used from many subclasses.
 */
class CCrossValidationOutput: public CSGObject
{
public:

	/** constructor */
	CCrossValidationOutput() : CSGObject()
	{
		m_current_run_index=0;
		m_current_fold_index=0;
		m_num_runs=0;
		m_num_folds=0;
	}

	/** destructor */
	virtual ~CCrossValidationOutput() {}

	/** @return name of SG_SERIALIZABLE */
	virtual const char* get_name() const=0;

	/** init number of runs (called once). saves them to local variable
	 *
	 * @param num_runs number of runs that will be performed
	 * @param prefix prefix for output
	 */
	virtual void init_num_runs(index_t num_runs, const char* prefix="")
	{
		m_num_runs=num_runs;
	}

	/** init number of folds (called once). saves them to local variable
	 * @param num_folds number of folds that will be performed
	 * @param prefix prefix for output
	 */
	virtual void init_num_folds(index_t num_folds, const char* prefix="")
	{
		m_num_folds=num_folds;
	}

	/** initially expose labels before usage
	 * @param labels labels to expose to CV output
	 */
	virtual void init_expose_labels(CLabels* labels) { }

	/** post init action (called once) */
	virtual void post_init() { }

	/** update run index (called every iteration). saves to local variable
	 *
	 * @param run_index index of current run
	 * @param prefix prefix for output
	 */
	virtual void update_run_index(index_t run_index,
			const char* prefix="")
	{
		m_current_run_index=run_index;
	}

	/** update fold index (called every iteration). saves to local variable
	 *
	 * @param fold_index index of current run
	 * @param prefix prefix for output
	 */
	virtual void update_fold_index(index_t fold_index,
			const char* prefix="")
	{
		m_current_fold_index=fold_index;
	}

	/** update train indices
	 *
	 * @param indices indices used for training
	 * @param prefix prefix for output
	 */
	virtual void update_train_indices(SGVector<index_t> indices,
			const char* prefix="") {}

	/** update test indices
	 *
	 * @param indices indices used for testing/validation
	 * @param prefix prefix for output
	 */
	virtual void update_test_indices(SGVector<index_t> indices,
			const char* prefix="") {}

	/** update trained machine
	 *
	 * @param machine trained machine instance
	 * @param prefix prefix for output
	 */
	virtual void update_trained_machine(CMachine* machine,
			const char* prefix="") {}

	/** update test result
	 *
	 * @param results result labels for test/validation run
	 * @param prefix prefix for output
	 */
	virtual void update_test_result(CLabels* results,
			const char* prefix="") {}

	/** update test true result
	 *
	 * @param results ground truth labels for test/validation run
	 * @param prefix prefix for output
	 */
	virtual void update_test_true_result(CLabels* results,
			const char* prefix="") {}

	/** post update test and true results 
	 */
	virtual void post_update_results() {}

	/** update evaluate result
	 *
	 * @param result evaluation result
	 * @param prefix prefix for output
	 */
	virtual void update_evaluation_result(float64_t result,
			const char* prefix="") {}

protected:
	/** current run index is written here */
	index_t m_current_run_index;

	/** current fold index is written here */
	index_t m_current_fold_index;

	/** number of runs is initialised here */
	index_t m_num_runs;

	/** number of folds is initialised here */
	index_t m_num_folds;
};

}

#endif /* __CROSSVALIDATIONOUTPUT_H_ */
