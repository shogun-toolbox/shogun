/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SVRLight_H___
#define _SVRLight_H___

#include <shogun/lib/config.h>
#include <shogun/regression/Regression.h>

#ifdef USE_SVMLIGHT
#include <shogun/classifier/svm/SVMLight.h>
#endif //USE_SVMLIGHT

#ifdef USE_SVMLIGHT
namespace shogun
{
/** @brief Class SVRLight, performs support vector regression using SVMLight.
 *
 * The SVR solution can be expressed as
 *  \f[
 *		f({\bf x})=\sum_{i=1}^{N} \alpha_i k({\bf x}, {\bf x_i})+b
 *	\f]
 *
 *	where \f$\alpha\f$ and \f$b\f$ are determined in training, i.e. using a
 *	pre-specified kernel, a given tube-epsilon for the epsilon insensitive
 *	loss, the follwoing quadratic problem is minimized (using the chunking
 *	decomposition technique)
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
 * This implementation supports multiple kernel learning, i.e. if a
 * CCombinedKernel is used the weights
 * \f$\beta\f$ in \f$ k_{combined}({\bf x}, {\bf x'}) = \sum_{m=0}^M \beta_m k_m({\bf x}, {\bf x'})\f$
 * can be determined in training (cf. Large Scale Multiple Kernel Learning
 * Sonnenburg, Raetsch, Schaefer, Schoelkopf 2006).
 *
 * \b linadd optimizations were implemented for kernels that support it (most
 * string kernels and the linear kernel), which will result in significant
 * speedups.
 */
class CSVRLight: public CSVMLight
{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);

		/** default constructor */
		CSVRLight();

		/** constructor
		 *
		 * @param C constant C
		 * @param epsilon epsilon
		 * @param k kernel
		 * @param lab labels
		 */
		CSVRLight(float64_t C, float64_t epsilon, CKernel* k, CLabels* lab);

		/** default destructor */
		virtual ~CSVRLight();

		/** get classifier type
		 *
		 * @return classifier type SVRLIGHT
		 */
		virtual EMachineType get_classifier_type();

		/** SVR learn */
		void   svr_learn();

		/** compute objective function
		 *
		 * @param a a
		 * @param lin lin
		 * @param c c
		 * @param eps eps
		 * @param label label
		 * @param totdoc totdoc
		 */
		virtual float64_t compute_objective_function(
			float64_t *a, float64_t *lin, float64_t *c, float64_t* eps,
			int32_t *label, int32_t totdoc);

		/** update linear component
		 *
		 * @param docs docs
		 * @param label label
		 * @param active2dnum active2dnum
		 * @param a a
		 * @param a_old a old
		 * @param working2dnum working2dnum
		 * @param totdoc totdoc
		 * @param lin lin
		 * @param aicache ai cache
		 * @param c c
		 */
		virtual void update_linear_component(
			int32_t* docs, int32_t *label,
			int32_t *active2dnum, float64_t *a, float64_t* a_old,
			int32_t *working2dnum, int32_t totdoc,
			float64_t *lin, float64_t *aicache, float64_t* c);

		/** update linear component MKL
		 *
		 * @param docs docs
		 * @param label label
		 * @param active2dnum active2dnum
		 * @param a a
		 * @param a_old a old
		 * @param working2dnum working2dnum
		 * @param totdoc totdoc
		 * @param lin lin
		 * @param aicache ai cache
		 * @param c c
		 */
		virtual void update_linear_component_mkl(
			int32_t* docs, int32_t *label,
			int32_t *active2dnum, float64_t *a, float64_t* a_old,
			int32_t *working2dnum, int32_t totdoc,
			float64_t *lin, float64_t *aicache, float64_t* c);

		/** update linear component MKL linadd
		 *
		 * @param docs docs
		 * @param label label
		 * @param active2dnum active2dnum
		 * @param a a
		 * @param a_old a old
		 * @param working2dnum working2dnum
		 * @param totdoc totdoc
		 * @param lin lin
		 * @param aicache ai cache
		 * @param c c
		 */
		virtual void update_linear_component_mkl_linadd(
			int32_t* docs, int32_t *label,
			int32_t *active2dnum, float64_t *a, float64_t* a_old,
			int32_t *working2dnum, int32_t totdoc,
			float64_t *lin, float64_t *aicache, float64_t* c);

		/** call mkl callback
		 * @param a
		 * @param label
		 * @param lin
		 * @param c
		 * @param totdoc
		 */
		void call_mkl_callback(float64_t* a, int32_t* label, float64_t* lin, float64_t* c, int32_t totdoc);

		/** reactivate inactive examples
		 *
		 * @param label label
		 * @param a a
		 * @param shrink_state shrink state
		 * @param lin lin
		 * @param c c
		 * @param totdoc totdoc
		 * @param iteration iteration
		 * @param inconsistent inconsistent
		 * @param docs docs
		 * @param aicache ai cache
		 * @param maxdiff maxdiff
		 */
		virtual void reactivate_inactive_examples(
			int32_t *label,float64_t *a,SHRINK_STATE *shrink_state,
			float64_t *lin, float64_t *c, int32_t totdoc,int32_t iteration,
			int32_t *inconsistent,
			int32_t *docs,float64_t *aicache,
			float64_t* maxdiff);

		/** @return object name */
		virtual const char* get_name() const { return "SVRLight"; }

	protected:
		/** thread helper for update linear component linadd
		 *
		 * @param params
		 */
		static void* update_linear_component_linadd_helper(void *params);

		/** regression fix index
		 *
		 * @param i i
		 * @return fix index
		 */
		int32_t regression_fix_index(int32_t i);

		/** regression fix index2
		 *
		 * @param i i
		 * @param num_vectors number of vectors
		 * @return fix index
		 */
		static int32_t regression_fix_index2(
			int32_t i, int32_t num_vectors);

		/** compute kernel at given index
		 *
		 * @param i index i
		 * @param j index j
		 * @return kernel value at i,j
		 */
		virtual float64_t compute_kernel(int32_t i, int32_t j);

		/** train regression
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based regressors are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

		/** number of train elements */
		int32_t num_vectors;
};
}
#endif //USE_SVMLIGHT
#endif
