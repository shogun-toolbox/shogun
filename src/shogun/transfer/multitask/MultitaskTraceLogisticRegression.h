/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  MULTITASKTRACELOGISTICREGRESSION_H_
#define  MULTITASKTRACELOGISTICREGRESSION_H_

#include <shogun/lib/config.h>

#include <shogun/transfer/multitask/MultitaskLogisticRegression.h>

namespace shogun
{
/** @brief class MultitaskTraceLogisticRegression, a classifier for multitask problems.
 * Supports only task group relations. Based on solver ported from the MALSAR library.
 *
 * @see CTaskGroup
 */
class CMultitaskTraceLogisticRegression : public CMultitaskLogisticRegression
{

	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY)

		/** default constructor */
		CMultitaskTraceLogisticRegression();

		/** constructor
		 *
		 * @param rho rho regularization coefficient
		 * @param training_data training features
		 * @param training_labels training labels
		 * @param task_relation task relation
		 */
		CMultitaskTraceLogisticRegression(
		     float64_t rho, CDotFeatures* training_data,
		     CBinaryLabels* training_labels, CTaskGroup* task_relation);

		/** destructor */
		virtual ~CMultitaskTraceLogisticRegression();

		/** set rho
		 * @param rho value
		 */
		void set_rho(float64_t rho);

		/** get rho
		 * @return rho value
		 */
		float64_t get_rho() const;

		/** get name
		 *
		 * @return name of the object
		 */
		virtual const char* get_name() const
		{
			return "MultitaskTraceLogisticRegression";
		}

	private:

		/** init */
		void init();

	protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data=NULL);

		/** train locked implementation */
		virtual bool train_locked_implementation(SGVector<index_t>* tasks);

	protected:

		/** rho */
		float64_t m_rho;

};
}
#endif
