/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBSVM_H___
#define _LIBSVM_H___

#include <shogun/lib/common.h>
#include <shogun/classifier/svm/SVM.h>
#include <shogun/classifier/svm/SVM_libsvm.h>

namespace shogun
{
enum LIBSVM_SOLVER_TYPE
{
	LIBSVM_C_SVC = 1,
	LIBSVM_NU_SVC = 2
};

/** @brief LibSVM */
class CLibSVM : public CSVM
{
	public:
		/** constructor */
		CLibSVM(LIBSVM_SOLVER_TYPE st=LIBSVM_C_SVC);

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		CLibSVM(float64_t C, CKernel* k, CLabels* lab);

		virtual ~CLibSVM();

		/** train SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/** get classifier type
		 *
		 * @return classifier type LIBSVM
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_LIBSVM; }

		/** @return object name */
		inline virtual const char* get_name() const { return "LibSVM"; }

	protected:
		/** SVM problem */
		svm_problem problem;
		/** SVM param */
		svm_parameter param;
		/** SVM model */
		struct svm_model* model;

		/** solver type */
		LIBSVM_SOLVER_TYPE solver_type;
};
}
#endif
