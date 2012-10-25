/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBSVM_MULTICLASS_H___
#define _LIBSVM_MULTICLASS_H___

#include <shogun/lib/common.h>
#include <shogun/multiclass/MulticlassSVM.h>
#include <shogun/lib/external/shogun_libsvm.h>
#include <shogun/classifier/svm/LibSVM.h>

namespace shogun
{
/** @brief class LibSVMMultiClass. Does one vs one
 * classification. */
class CMulticlassLibSVM : public CMulticlassSVM
{
	public:
		/** default constructor */
		CMulticlassLibSVM(LIBSVM_SOLVER_TYPE st=LIBSVM_C_SVC);

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		CMulticlassLibSVM(float64_t C, CKernel* k, CLabels* lab);

		/** destructor */
		virtual ~CMulticlassLibSVM();

		/** get classifier type
		 *
		 * @return classifier type LIBSVMMULTICLASS
		 */
		virtual EMachineType get_classifier_type() { return CT_LIBSVMMULTICLASS; }

		/** @return object name */
		virtual const char* get_name() const { return "MulticlassLibSVM"; }

	protected:
		/** train multiclass SVM classifier
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

		/** solver type */
		LIBSVM_SOLVER_TYPE solver_type;
};
}
#endif
