/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  MULTITASKL1L2LOGISTICREGRESSION_H_
#define  MULTITASKL1L2LOGISTICREGRESSION_H_

#include <shogun/transfer/multitask/MultitaskLogisticRegression.h>

namespace shogun
{
/** @brief  */
class CMultitaskL1L2LogisticRegression : public CMultitaskLogisticRegression
{

	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY)

		/** default constructor */
		CMultitaskL1L2LogisticRegression();

		/** constructor
		 *
		 * @param rho1 rho1 regularization coefficient
		 * @param rho2 rho2 regularization coefficient
		 * @param training_data training features
		 * @param training_labels training labels
		 * @param task_relation task relation
		 */
		CMultitaskL1L2LogisticRegression(
		     float64_t rho1, float64_t rho2, CDotFeatures* training_data, 
		     CBinaryLabels* training_labels, CTaskGroup* task_group);

		/** destructor */
		virtual ~CMultitaskL1L2LogisticRegression();

		/** set rho1
		 * @param rho1 value
		 */
		void set_rho1(float64_t rho1); 
		
		/** set rho1
		 * @param rho2 value
		 */
		void set_rho2(float64_t rho2); 

		/** get name */
		virtual const char* get_name() const 
		{
			return "MultitaskL1L2LogisticRegression";
		}

	private:

		void init();

	protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data=NULL);
		
		/** train locked implementation */
		virtual bool train_locked_implementation(SGVector<index_t> indices,
		                                         SGVector<index_t>* tasks);

	protected:

		/** rho1 */
		float64_t m_rho1;
		
		/** rho2 */
		float64_t m_rho2;

};
}
#endif
