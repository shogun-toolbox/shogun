/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef _SVMLIGHTONECLASS_H___
#define _SVMLIGHTONECLASS_H___

#include "lib/config.h"
#include "classifier/Classifier.h"

#ifdef USE_SVMLIGHT
#include "classifier/svm/SVMLight.h"
#endif //USE_SVMLIGHT

#ifdef USE_SVMLIGHT
namespace shogun
{
/** @brief Trains a one class C SVM
 *
 * \sa CSVMLight
 */
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class CSVMLightOneClass: public CSVMLight
{
	public:
		/** default constructor */
		CSVMLightOneClass();

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 */
		CSVMLightOneClass(float64_t C, CKernel* k);

		/** default destructor */
		virtual ~CSVMLightOneClass() { }

		/** train one class svm
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based regressors are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/** get classifier type
		 *
		 * @return classifier type LIGHTONECLASS
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_LIGHTONECLASS; }
};
}
#endif //USE_SVMLIGHT
#endif // _SVMLIGHTONECLASS_H___
