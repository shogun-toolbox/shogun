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
#include <shogun/transfer/multitask/MultitaskLinearMachine.h>
#include <shogun/transfer/multitask/TaskRelation.h>
#include <shogun/transfer/multitask/TaskGroup.h>
#include <shogun/transfer/multitask/TaskTree.h>
#include <shogun/transfer/multitask/Task.h>


using namespace std;

namespace shogun
{
/** @brief class Multitask Logistic Regression used
 * to solve classification problems with a few tasks
 * related via group or tree. Based on L1/Lq regression
 * for groups and L1/L2 for trees.
 *
 * The underlying solver is based on the SLEP library.
 *
 */
class CMultitaskLogisticRegression : public CMultitaskLinearMachine
{

	public:
		/** problem type */
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

		/** get max iter */
		int32_t get_max_iter() const;
		/** get q */
		float64_t get_q() const;
		/** get regularization */
		int32_t get_regularization() const;
		/** get termination */
		int32_t get_termination() const;
		/** get tolerance */
		float64_t get_tolerance() const;
		/** get z */
		float64_t get_z() const;

		/** set max iter */
		void set_max_iter(int32_t max_iter);
		/** set q */
		void set_q(float64_t q);
		/** set regularization */
		void set_regularization(int32_t regularization);
		/** set termination */
		void set_termination(int32_t termination);
		/** set tolerance */
		void set_tolerance(float64_t tolerance);
		/** set z */
		void set_z(float64_t z);

		/** applies to one vector */
		virtual float64_t apply_one(int32_t i);

	protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data=NULL);

		/** train locked implementation */
		virtual bool train_locked_implementation(SGVector<index_t>* tasks);

	private:

		/** register parameters */
		void register_parameters();

		/** initialize parameters */
		void initialize_parameters();

	protected:

		/** regularization type */
		int32_t m_regularization;

		/** termination criteria */
		int32_t m_termination;

		/** max iteration */
		int32_t m_max_iter;

		/** tolerance */
		float64_t m_tolerance;

		/** q of L1/Lq */
		float64_t m_q;

		/** regularization coefficient */
		float64_t m_z;

};
}
#endif
