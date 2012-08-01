/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  MULTITASKLOGISTICREGRESSION_H_
#define  MULTITASKLOGISTICREGRESSION_H_

#include <shogun/lib/config.h>
#include <shogun/machine/SLEPMachine.h>
#include <shogun/transfer/multitask/TaskRelation.h>
#include <shogun/transfer/multitask/TaskGroup.h>
#include <shogun/transfer/multitask/TaskTree.h>
#include <shogun/transfer/multitask/Task.h>

#include <vector>
#include <set>

using namespace std;

namespace shogun
{
/** @brief  */
class CMultitaskLogisticRegression : public CSLEPMachine
{

	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY)

		/** default constructor */
		CMultitaskLogisticRegression();

		/** constructor
		 *
		 * @param z regularization coefficient
		 * @param training_data training features
		 * @param training_labels training labels
		 * @param task_relation task relation
		 */
		CMultitaskLogisticRegression(
		     float64_t z, CDotFeatures* training_data, 
		     CBinaryLabels* training_labels, CTaskRelation* task_relation);

		/** destructor */
		virtual ~CMultitaskLogisticRegression();

		/** get name */
		virtual const char* get_name() const 
		{
			return "MultitaskLogisticRegression";
		}

		/** getter for current task
		 * @return current task index
		 */
		int32_t get_current_task() const;

		/** setter for current task
		 * @param task task index
		 */
		void set_current_task(int32_t task);
		
		/** get w
		 *
		 * @return weight vector
		 */
		virtual SGVector<float64_t> get_w() const;

		/** set w
		 *
		 * @param src_w new w
		 */
		virtual void set_w(const SGVector<float64_t> src_w);

		/** set bias
		 *
		 * @param b new bias
		 */
		virtual void set_bias(float64_t b);

		/** get bias
		 *
		 * @return bias
		 */
		virtual float64_t get_bias();

		/** getter for task tree
		 * @return task tree
		 */
		CTaskRelation* get_task_relation() const;

		/** setter for task tree
		 * @param task_tree task tree
		 */
		void set_task_relation(CTaskRelation* task_relation);

		/** @return whether machine supports locking */
		virtual bool supports_locking() const { return true; }

		/** post lock */
		virtual void post_lock();

		/** train on given indices */
		virtual bool train_locked(SGVector<index_t> indices);

		/** applies on given indices */
		virtual CBinaryLabels* apply_locked_binary(SGVector<index_t> indices);

		/** applies to one vector */
		float64_t apply_one(int32_t i);

	protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data=NULL);

		/** train locked implementation */
		virtual bool train_locked_implementation(SGVector<index_t> indices, SGVector<index_t>* tasks);

		/** subset mapped task indices */
		SGVector<index_t>* get_subset_tasks_indices();

	private:

		/** register parameters */
		void register_parameters();

	protected:

		/** current task index */
		int32_t m_current_task;

		/** feature tree */
		CTaskRelation* m_task_relation;

		/** tasks w's */
		SGMatrix<float64_t> m_tasks_w;
		
		/** tasks interceptss */
		SGVector<float64_t> m_tasks_c;

		/** vector of sets of indices */
		vector< set<index_t> > m_tasks_indices;

};
}
#endif
