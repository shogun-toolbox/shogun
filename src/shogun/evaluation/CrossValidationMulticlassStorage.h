/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann, Sergey Lisitsyn
 *
 */

#ifndef CROSSVALIDATIONMULTICLASSSTORAGE_H_
#define CROSSVALIDATIONMULTICLASSSTORAGE_H_

#include <shogun/evaluation/CrossValidationOutput.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

class CMachine;
class CLabels;
class CEvaluation;

/** @brief Class for storing multiclass evaluation information in every fold of cross-validation */
class CCrossValidationMulticlassStorage: public CCrossValidationOutput
{
public:

	/** constructor */
	CCrossValidationMulticlassStorage() : CCrossValidationOutput()
	{
		m_pred_labels = NULL;
		m_true_labels = NULL;
		m_num_classes = 0;
	}

	/** destructor */
	virtual ~CCrossValidationMulticlassStorage()
	{
		for (int32_t i=0; i<m_num_folds*m_num_runs; i++)
		{
			m_fold_ROC_graphs[i].~SGMatrix<float64_t>();
		}
		SG_FREE(m_fold_ROC_graphs);
	};

	/** returns ROC
	 * 
	 * @param run run
	 * @param fold fold
	 * @param c class
	 * @return ROC of 'run' run, 'fold' fold and 'c' class
	 */
	SGMatrix<float64_t> get_fold_ROC(int32_t run, int32_t fold, int32_t c)
	{
		ASSERT(0<=run);
		ASSERT(run<m_num_runs);
		ASSERT(0<=fold);
		ASSERT(fold<m_num_folds);
		ASSERT(0<=c);
		ASSERT(c<m_num_classes);
		return m_fold_ROC_graphs[run*m_num_folds*m_num_classes+fold*m_num_classes+c];
	}

	/** post init */
	virtual void post_init();

	/** post update results */
	virtual void post_update_results();

	/** expose labels 
	 * @param labels labels to expose
	 */
	virtual void init_expose_labels(CLabels* labels);

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

	/** @return name of SG_SERIALIZABLE */
	virtual const char* get_name() const { return "CrossValidationMulticlassStorage"; }

protected:

	/** fold ROC graphs */
	SGMatrix<float64_t>* m_fold_ROC_graphs; 

	/** predicted results */
	CMulticlassLabels* m_pred_labels;

	/** true labels */
	CMulticlassLabels* m_true_labels;

	/** number of classes */
	int32_t m_num_classes;

};

}

#endif /* CROSSVALIDATIONMULTICLASSSTORAGE_H_ */
