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

#ifndef __CROSSVALIDATIONPRINTOUTPUT_H_
#define __CROSSVALIDATIONPRINTOUTPUT_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/CrossValidationOutput.h>

namespace shogun
{

class CMachine;
class CLabels;
class CEvaluation;

/** @brief Class for outputting cross-validation intermediate results to the
 * standard output. Simply prints all messages it gets */
class CCrossValidationPrintOutput: public CCrossValidationOutput
{
public:

	/** constructor */
	CCrossValidationPrintOutput() {};

	/** destructor */
	virtual ~CCrossValidationPrintOutput() {};

	/** @return name of SG_SERIALIZABLE */
	virtual const char* get_name() const { return "CrossValidationPrintOutput"; }

	/** init number of runs (called once)
	 *
	 * @param num_runs number of runs that will be performed
	 * @param prefix prefix for output
	 */
	virtual void init_num_runs(index_t num_runs, const char* prefix="");

	/** init number of folds
	 * @param num_folds number of folds that will be performed
	 * @param prefix prefix for output
	 */
	virtual void init_num_folds(index_t num_folds, const char* prefix="");

	/** update run index
	 *
	 * @param run_index index of current run
	 * @param prefix prefix for output
	 */
	virtual void update_run_index(index_t run_index,
			const char* prefix="");

	/** update fold index
	 *
	 * @param fold_index index of current run
	 * @param prefix prefix for output
	 */
	virtual void update_fold_index(index_t fold_index,
			const char* prefix="");

	/** update train indices
	 *
	 * @param indices indices used for training
	 * @param prefix prefix for output
	 */
	virtual void update_train_indices(SGVector<index_t> indices,
			const char* prefix="");

	/** update test indices
	 *
	 * @param indices indices used for testing/validation
	 * @param prefix prefix for output
	 */
	virtual void update_test_indices(SGVector<index_t> indices,
			const char* prefix="");

	/** update trained machine
	 *
	 * @param machine trained machine instance
	 * @param prefix prefix for output
	 */
	virtual void update_trained_machine(CMachine* machine,
			const char* prefix="");

	/** update test result
	 *
	 * @param results result labels for test/validation run
	 * @param prefix prefix for output
	 */
	virtual void update_test_result(CLabels* results,
			const char* prefix="");

	/** update test true result
	 *
	 * @param results ground truth labels for test/validation run
	 * @param prefix prefix for output
	 */
	virtual void update_test_true_result(CLabels* results,
			const char* prefix="");

	/** update evaluate result
	 *
	 * @param result evaluation result
	 * @param prefix prefix for output
	 */
	virtual void update_evaluation_result(float64_t result,
			const char* prefix="");

protected:
	/** returns a string which is the provided one plus a tab character
	 *
	 * @param string null-terminated string to append tab to
	 * @return null-terminated string with tab appended
	 */
	char* append_tab_to_string(const char* string);
};

}

#endif /* __CROSSVALIDATIONPRINTOUTPUT_H_ */
