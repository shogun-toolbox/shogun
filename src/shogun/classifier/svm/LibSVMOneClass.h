/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 2006-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBSVM_ONECLASS_H___
#define _LIBSVM_ONECLASS_H___
#include <lib/common.h>
#include <classifier/svm/SVM.h>
#include <lib/external/shogun_libsvm.h>

#include <stdio.h>

namespace shogun
{
/** @brief class LibSVMOneClass */
class CLibSVMOneClass : public CSVM
{
	public:
		/** default constructor */
		CLibSVMOneClass();

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 */
		CLibSVMOneClass(float64_t C, CKernel* k);
		virtual ~CLibSVMOneClass();

		/** get classifier type
		 *
		 * @return classifier type LIBSVMONECLASS
		 */
		virtual EMachineType get_classifier_type() { return CT_LIBSVMONECLASS; }

		/** @return object name */
		virtual const char* get_name() const { return "LibSVMOneClass"; }

	protected:

		virtual bool train_require_labels() const { return false; }

		/** train SVM
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

	protected:
		/** SVM problem */
		svm_problem problem;
		/** SVM parameter */
		svm_parameter param;

		/** SVM model */
		struct svm_model* model;
};
}
#endif
