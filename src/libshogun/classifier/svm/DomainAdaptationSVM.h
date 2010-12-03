/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Christian Widmer
 * Copyright (C) 2007-2009 Max-Planck-Society
 */

#ifndef _DomainAdaptation_SVM_H___
#define _DomainAdaptation_SVM_H___


#include "lib/common.h"
#include "classifier/svm/SVMLight.h"

#include <stdio.h>

namespace shogun
{
/** @brief class DomainAdaptiveSVM */
class CDomainAdaptationSVM : public CSVMLight
{
	public:

		/** default constructor */
		CDomainAdaptationSVM();

		/** constructor
		 *
		 * @param C cost constant C
		 * @param k kernel
		 * @param lab labels
		 * @param presvm trained SVM to regularize against
		 * @param B trade-off constant B
		 */
		CDomainAdaptationSVM(float64_t C, CKernel* k, CLabels* lab, CSVM* presvm, float64_t B);

		/** destructor */
		virtual ~CDomainAdaptationSVM();

		/** init SVM
		 *
		 * @param presvm trained SVM to regularize against
		 * @param B trade-off constant B
		 * */
		void init(CSVM* presvm, float64_t B);

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
		 * @return classifier type LIGHT
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_DASVM; }

		/** classify objects
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CLabels* classify(CFeatures* data);

		/** returns SVM that is used as prior information
		 *
		 * @return presvm
		 */
		virtual CSVM* get_presvm();

		/** getter for regularization parameter B
		 *
		 * @return regularization parameter B
		 */
		virtual float64_t get_B();

		/** getter for train_factor
		 *
		 * @return train_factor
		 */
		virtual float64_t get_train_factor();

		/** setter for train_factor
		 *
		 */
		virtual void set_train_factor(float64_t factor);

		/** @return object name */
		inline virtual const char* get_name() const { return "DomainAdaptationSVM"; }

	protected:

		/** check sanity of presvm
		 *
		 * @return true if sane, throws SG_ERROR otherwise
		 */
		virtual bool is_presvm_sane();

	private:
		void init();

	protected:

		/** SVM to regularize against */
		CSVM* presvm;

		/** regularization parameter B */
		float64_t B;

		/** flag to switch off regularization in training */
		float64_t train_factor;
};
}
#endif
