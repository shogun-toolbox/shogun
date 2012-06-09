/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  MULTITASKLEASTSQUARESREGRESSION_H_
#define  MULTITASKLEASTSQUARESREGRESSION_H_

#include <shogun/lib/config.h>
#include <shogun/lib/IndicesTree.h>
#include <shogun/machine/SLEPMachine.h>

namespace shogun
{
/** @brief  */
class CMultitaskLeastSquaresRegression : public CSLEPMachine
{

	public:
		MACHINE_PROBLEM_TYPE(PT_REGRESSION)

		/** default constructor */
		CMultitaskLeastSquaresRegression();

		/** constructor
		 *
		 * @param z regularization coefficient
		 * @param training_data training features
		 * @param training_labels training labels
		 * @param task_tree task tree 
		 */
		CMultitaskLeastSquaresRegression(
		     float64_t z, CDotFeatures* training_data, 
		     CRegressionLabels* training_labels, CIndicesTree* task_tree);

		/** destructor */
		virtual ~CMultitaskLeastSquaresRegression();

		/** get name */
		virtual const char* get_name() const 
		{
			return "MultitaskLeastSquaresRegression";
		}

		/** getter for current task
		 * @return current task index
		 */
		int32_t get_current_task() const;

		/** setter for current task
		 * @param task task index
		 */
		void set_current_task(int32_t task);

		/** getter for task tree
		 * @return task tree
		 */
		CIndicesTree* get_task_tree() const;

		/** setter for task tree
		 * @param task_tree task tree
		 */
		void set_task_tree(CIndicesTree* task_tree);
		
	protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data=NULL);

	private:

		/** register parameters */
		void register_parameters();

	protected:

		/** current task index */
		int32_t m_current_task;

		/** feature tree */
		CIndicesTree* m_task_tree;

		/** tasks w's */
		SGMatrix<float64_t> m_tasks_w;

};
}
#endif   /* ----- #ifndef MULTITASKLEASTSQUARESREGRESSION_H_  ----- */
