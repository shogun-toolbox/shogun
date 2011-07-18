/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBSVR_H___
#define _LIBSVR_H___

#include <stdio.h>

#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "classifier/svm/SVM_libsvm.h"
#include "regression/Regression.h"

namespace shogun
{
/** @brief Class LibSVR, performs support vector regression using LibSVM.
 *
 * The SVR solution can be expressed as 
 *  \f[
 * 		f({\bf x})=\sum_{i=1}^{N} \alpha_i k({\bf x}, {\bf x_i})+b
 * 	\f]
 *
 * 	where \f$\alpha\f$ and \f$b\f$ are determined in training, i.e. using a
 * 	pre-specified kernel, a given tube-epsilon for the epsilon insensitive
 * 	loss, the follwoing quadratic problem is minimized (using sequential
 * 	minimal decomposition (SMO))
 *
 * 	\f{eqnarray*}
 * 		\max_{{\bf \alpha},{\bf \alpha}^*} &-\frac{1}{2}\sum_{i,j=1}^N(\alpha_i-\alpha_i^*)(\alpha_j-\alpha_j^*){\bf x}_i^T {\bf x}_j -\sum_{i=1}^N(\alpha_i+\alpha_i^*)\epsilon - \sum_{i=1}^N(\alpha_i-\alpha_i^*)y_i\\
 * 		\mbox{wrt}:& {\bf \alpha},{\bf \alpha}^*\in{\bf R}^N\\
 * 		\mbox{s.t.}:& 0\leq \alpha_i,\alpha_i^*\leq C,\, \forall i=1\dots N\\
 * 					&\sum_{i=1}^N(\alpha_i-\alpha_i^*)y_i=0
 * \f}
 *
 *
 * Note that the SV regression problem is reduced to the standard SV
 * classification problem by introducing artificial labels \f$-y_i\f$ which
 * leads to the epsilon insensitive loss constraints * \f{eqnarray*}
 * 		{\bf w}^T{\bf x}_i+b-c_i-\xi_i\leq 0,&\, \forall i=1\dots N\\
 * 		-{\bf w}^T{\bf x}_i-b-c_i^*-\xi_i^*\leq 0,&\, \forall i=1\dots N
 * \f}
 * with \f$c_i=y_i+ \epsilon\f$ and \f$c_i^*=-y_i+ \epsilon\f$
 */
class CLibSVR : public CSVM
{
	public:
		/** default constructor */
		CLibSVR();

		/** constructor
		 *
		 * @param C constant C
		 * @param epsilon tube epsilon
		 * @param k kernel
		 * @param lab labels
		 */
		CLibSVR(float64_t C, float64_t epsilon, CKernel* k, CLabels* lab);
		virtual ~CLibSVR();

		/** train regression
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based regressor are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/** get classifier type
		 *
		 * @return classifie type LIBSVR
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_LIBSVR; }

		/** @return object name */
		inline virtual const char* get_name() const { return "LibSVR"; }
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
