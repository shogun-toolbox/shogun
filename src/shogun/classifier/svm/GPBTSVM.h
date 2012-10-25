/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _GPBTSVM_H___
#define _GPBTSVM_H___
#include <shogun/lib/common.h>
#include <shogun/classifier/svm/SVM.h>
#include <shogun/lib/external/shogun_libsvm.h>

#include <stdio.h>

namespace shogun
{
/** @brief class GPBTSVM */
class CGPBTSVM : public CSVM
{
	public:
		/** default constructor */
		CGPBTSVM();

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		CGPBTSVM(float64_t C, CKernel* k, CLabels* lab);
		virtual ~CGPBTSVM();

		/** get classifier type
		 *
		 * @return classifier type GPBT
		 */
		virtual EMachineType get_classifier_type() { return CT_GPBT; }

		/** @return object name */
		virtual const char* get_name() const { return "GPBTSVM"; }

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

	protected:
		/** SVM model */
		struct svm_model* model;
};
}
#endif
