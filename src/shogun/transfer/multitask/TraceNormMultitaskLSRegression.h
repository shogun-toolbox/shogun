/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  TRACENORMMULTITASKLSREGRESSION_H_
#define  TRACENORMMULTITASKLSREGRESSION_H_

#include <shogun/transfer/multitask/MultitaskLSRegression.h>
#include <shogun/lib/IndexBlockGroup.h>

namespace shogun
{
/** @brief  */
class CTraceNormMultitaskLSRegression : public CMultitaskLSRegression
{

	public:
		MACHINE_PROBLEM_TYPE(PT_REGRESSION)

		/** default constructor */
		CTraceNormMultitaskLSRegression();

		/** constructor
		 *
		 * @param z regularization coefficient
		 * @param training_data training features
		 * @param training_labels training labels
		 * @param task_group task group
		 */
		CTraceNormMultitaskLSRegression(
		     float64_t z, CDotFeatures* training_data, 
		     CRegressionLabels* training_labels, CIndexBlockGroup* task_group);

		/** destructor */
		virtual ~CTraceNormMultitaskLSRegression();

		/** get name */
		virtual const char* get_name() const 
		{
			return "TraceNormMultitaskLSRegression";
		}

	protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data=NULL);

};
}
#endif
