/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  MULTITASKCOMPOSITEMACHINE_H_
#define  MULTITASKCOMPOSITEMACHINE_H_

#include <shogun/lib/config.h>
#include <shogun/machine/Machine.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/transfer/multitask/TaskRelation.h>
#include <shogun/transfer/multitask/TaskGroup.h>
#include <shogun/transfer/multitask/TaskTree.h>
#include <shogun/transfer/multitask/Task.h>

#include <vector>
#include <set>

using namespace std;

namespace shogun
{
/** @brief class MultitaskCompositeMachine used to
 * solve multitask binary classification problems with separate
 * training of given binary classifier on each task
 */
class CMultitaskCompositeMachine : public CMachine
{

	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY)

		/** default constructor */
		CMultitaskCompositeMachine();

		/** constructor
		 *
		 * @param machine 
		 * @param training_data training features
		 * @param training_labels training labels
		 * @param task_group task group
		 */
		CMultitaskCompositeMachine(
		     CMachine* machine, CFeatures* training_data, 
		     CLabels* training_labels, CTaskGroup* task_group);

		/** destructor */
		virtual ~CMultitaskCompositeMachine();

		/** get name */
		virtual const char* get_name() const 
		{
			return "MultitaskCompositeMachine";
		}

		/** getter for current task
		 * @return current task index
		 */
		int32_t get_current_task() const;

		/** setter for current task
		 * @param task task index
		 */
		void set_current_task(int32_t task);

		/** getter for task group
		 * @return task group
		 */
		CTaskGroup* get_task_group() const;

		/** setter for task group
		 * @param task_group task group
		 */
		void set_task_group(CTaskGroup* task_group);

		/** @return whether machine supports locking */
		virtual bool supports_locking() const { return true; }

		/** post lock */
		virtual void post_lock(CLabels* labels, CFeatures* features);

		/** train on given indices */
		virtual bool train_locked(SGVector<index_t> indices);

		/** applies on given indices */
		virtual CBinaryLabels* apply_locked_binary(SGVector<index_t> indices);

		/** set features
		 *
		 * @param features features to set
		 */
		virtual void set_features(CFeatures* features)
		{
			SG_REF(features);
			SG_UNREF(m_features);
			m_features = features;
		}

		/** get features
		 *
		 * @return features
		 */
		virtual CFeatures* get_features() const
		{ 
			SG_REF(m_features);
			return m_features;
		}
		
		/** set machine
		 *
		 * @param machine machine
		 */
		virtual void set_machine(CMachine* machine)
		{
			SG_REF(machine);
			SG_UNREF(m_machine);
			m_machine = machine;
		}

		/** get machine
		 *
		 * @return machine
		 */
		virtual CMachine* get_machine() const
		{ 
			SG_REF(m_machine);
			return m_machine;
		}

		/** applies to one vector */
		virtual float64_t apply_one(int32_t vec_idx);

	protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data=NULL);

	private:

		/** register parameters */
		void register_parameters();

	protected:

		/** machine */
		CMachine* m_machine;

		/** features */
		CFeatures* m_features;

		/** current task index */
		int32_t m_current_task;

		/** feature tree */
		CTaskGroup* m_task_group;

		/** trained machines*/
		CDynamicObjectArray* m_task_machines;

		/** tasks indices */
		vector< set<index_t> > m_tasks_indices;

};
}
#endif
