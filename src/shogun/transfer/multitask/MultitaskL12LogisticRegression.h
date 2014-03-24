/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  MULTITASKL12LOGISTICREGRESSION_H_
#define  MULTITASKL12LOGISTICREGRESSION_H_

#include <shogun/lib/config.h>
#include <shogun/transfer/multitask/MultitaskLogisticRegression.h>

namespace shogun
{
/** @brief class MultitaskL12LogisticRegression, a classifier for multitask problems.
 * Supports only task group relations. Based on solver ported from the MALSAR library.
 *
 * @see CTaskGroup
 * */
class CMultitaskL12LogisticRegression : public CMultitaskLogisticRegression
{

	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY)

		/** default constructor */
		CMultitaskL12LogisticRegression();

		/** constructor
		 *
		 * @param rho1 rho1 regularization coefficient of L1/L2 term
		 * @param rho2 rho2 regularization coefficient of L2 term
		 * @param training_data training features
		 * @param training_labels training labels
		 * @param task_group task group
		 */
		CMultitaskL12LogisticRegression(
		     float64_t rho1, float64_t rho2, CDotFeatures* training_data,
		     CBinaryLabels* training_labels, CTaskGroup* task_group);

		/** destructor */
		virtual ~CMultitaskL12LogisticRegression();

		/** set rho1 regularization coefficient
		 * @param rho1 value
		 */
		void set_rho1(float64_t rho1);

		/** get rho1 regularization coefficient
		 * @return rho1 value
		 */
		float64_t get_rho1() const;

		/** set rho2 regularization coefficient
		 * @param rho2 value
		 */
		void set_rho2(float64_t rho2);

		/** get rho2 regularization coefficient
		 * @return rho2 value
		 */
		float64_t get_rho2() const;

		/** get name
		 *
		 * @return name of the object
		 */
		virtual const char* get_name() const
		{
			return "MultitaskL12LogisticRegression";
		}

	private:

		/** init */
		void init();

	protected:

		/** train machine
		 *
		 * @param data features to use for training
		 */
		virtual bool train_machine(CFeatures* data=NULL);

		/** train locked implementation
		 *
		 * @param tasks array of tasks indices
		 */
		virtual bool train_locked_implementation(SGVector<index_t>* tasks);

	protected:

		/** rho1, regularization coefficient of L1/L2 term */
		float64_t m_rho1;

		/** rho2, regularization coefficient of L2 term */
		float64_t m_rho2;

};
}
#endif
