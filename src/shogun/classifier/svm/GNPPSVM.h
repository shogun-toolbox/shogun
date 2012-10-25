/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2008 Center for Machine Perception, CTU FEL Prague
 */

#ifndef _GNPPSVM_H___
#define _GNPPSVM_H___

#include <shogun/lib/common.h>
#include <shogun/classifier/svm/SVM.h>

namespace shogun
{
/** @brief class GNPPSVM */
class CGNPPSVM : public CSVM
{
	public:
		/** default constructor */
		CGNPPSVM();

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		CGNPPSVM(float64_t C, CKernel* k, CLabels* lab);

		virtual ~CGNPPSVM();

		/** get classifier type
		 *
		 * @return classifier type GNPPSVM
		 */
		virtual EMachineType get_classifier_type() { return CT_GNPPSVM; }

		/** @return object name */
		virtual const char* get_name() const { return "GNPPSVM"; }

	protected:
		/** train SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);
};
}
#endif
