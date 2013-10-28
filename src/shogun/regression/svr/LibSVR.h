/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2013 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBSVR_H___
#define _LIBSVR_H___

#include <stdio.h>

#include <shogun/lib/common.h>
#include <shogun/classifier/svm/SVM.h>
#include <shogun/lib/external/shogun_libsvm.h>
#include <shogun/regression/Regression.h>

namespace shogun
{
/** @brief Class LibSVR, performs support vector regression using LibSVM.
 *
 * The SVR solution can be expressed as
 *  \f[
 *		f({\bf x})=\sum_{i=1}^{N} \alpha_i k({\bf x}, {\bf x_i})+b
 *	\f]
 *
 *	where \f$\alpha\f$ and \f$b\f$ are determined in training, i.e. using a
 *	pre-specified kernel, a given tube-epsilon for the epsilon insensitive
 *	loss, the follwoing quadratic problem is minimized (using sequential
 *	minimal decomposition (SMO))
 *
 *	\f{eqnarray*}
 *		\max_{{\bf \alpha},{\bf \alpha}^*} &-\frac{1}{2}\sum_{i,j=1}^N(\alpha_i-\alpha_i^*)(\alpha_j-\alpha_j^*){\bf x}_i^T {\bf x}_j -\sum_{i=1}^N(\alpha_i+\alpha_i^*)\epsilon - \sum_{i=1}^N(\alpha_i-\alpha_i^*)y_i\\
 *		\mbox{wrt}:& {\bf \alpha},{\bf \alpha}^*\in{\bf R}^N\\
 *		\mbox{s.t.}:& 0\leq \alpha_i,\alpha_i^*\leq C,\, \forall i=1\dots N\\
 *					&\sum_{i=1}^N(\alpha_i-\alpha_i^*)y_i=0
 * \f}
 *
 *
 * Note that the SV regression problem is reduced to the standard SV
 * classification problem by introducing artificial labels \f$-y_i\f$ which
 * leads to the epsilon insensitive loss constraints * \f{eqnarray*}
 *		{\bf w}^T{\bf x}_i+b-c_i-\xi_i\leq 0,&\, \forall i=1\dots N\\
 *		-{\bf w}^T{\bf x}_i-b-c_i^*-\xi_i^*\leq 0,&\, \forall i=1\dots N
 * \f}
 * with \f$c_i=y_i+ \epsilon\f$ and \f$c_i^*=-y_i+ \epsilon\f$
 *
 * This class also support the \f$\nu\f$-SVR regression version of the problem,
 * where \f$\nu\f$ replaces the \f$\epsilon\f$ parameter and represents an
 * upper bound on the fraction of margin errors and a lower bound on the fraction
 * of support vectors. While it is easier to interpret, the resulting
 * optimization problem usually takes longer to solve. Note that these different
 * parameters do not result in different predictive power. For a given problem,
 * the best SVR for each parametrization will lead to the same results.
 * See the letter "Training \f$\nu\f$-Support Vector Regression: Theory and Algorithms" by
 * Chih-Chung Chang and Chih-Jen Lin for the relation of \f$\epsilon\f$-SVR and
 * \f$\nu\f$-SVR.
 */
#ifndef DOXYGEN_SHOULD_SKIP_THIS
enum LIBSVR_SOLVER_TYPE
{
	LIBSVR_EPSILON_SVR = 1,
	LIBSVR_NU_SVR = 2
};
#endif
class CLibSVR : public CSVM
{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);

		/** default constructor, creates a EPISOLON-SVR */
		CLibSVR();

		/** constructor
		 *
		 * @param C constant C
		 * @param svr_param tube epsilon or SVR-NU depending on solver type
		 * @param k kernel
		 * @param lab labels
		 * @param st solver type to use, EPSILON-SVR or NU-SVR
		 */
		CLibSVR(float64_t C, float64_t svr_param, CKernel* k, CLabels* lab,
				LIBSVR_SOLVER_TYPE st=LIBSVR_EPSILON_SVR);

		virtual ~CLibSVR();

		/** get classifier type
		 *
		 * @return classifie type LIBSVR
		 */
		virtual EMachineType get_classifier_type();

		/** @return object name */
		virtual const char* get_name() const { return "LibSVR"; }

	protected:
		/** train regression
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based regressor are used and distance/kernels are
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
		LIBSVR_SOLVER_TYPE solver_type;
};
}
#endif
