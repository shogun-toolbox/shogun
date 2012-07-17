/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  L1LOGISTICREGRESSION_H_
#define  L1LOGISTICREGRESSION_H_

#include <shogun/lib/config.h>
#include <shogun/machine/SLEPMachine.h>

namespace shogun
{
/** @brief  */
class CL1LogisticRegression : public CSLEPMachine
{

	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY)

		/** default constructor */
		CL1LogisticRegression();

		/** constructor
		 *
		 * @param z regularization coefficient
		 * @param training_data training features
		 * @param training_labels training labels
		 */
		CL1LogisticRegression(
		     float64_t z, CDotFeatures* training_data, 
		     CBinaryLabels* training_labels);

		/** destructor */
		virtual ~CL1LogisticRegression();

		/** get name */
		virtual const char* get_name() const 
		{
			return "L1LogisticRegression";
		}

		virtual float64_t apply_one(int32_t vec_idx);

	protected:
		
		virtual SGVector<float64_t> apply_get_outputs(CFeatures* data);

		/** train machine */
		virtual bool train_machine(CFeatures* data=NULL);

};
}
#endif
