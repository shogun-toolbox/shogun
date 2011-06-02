/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Mikio L. Braun
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KRR_H__
#define _KRR_H__

#include "lib/config.h"
#include "regression/Regression.h"

#ifdef HAVE_LAPACK

#include "machine/KernelMachine.h"

namespace shogun
{
/** @brief Class KRR implements Kernel Ridge Regression - a regularized least square
 * method for classification and regression.
 *
 * It is similar to support vector machines (cf. CSVM). However in contrast to
 * SVMs a different objective is optimized that leads to a dense solution (thus
 * not only a few support vectors are active in the end but all training
 * examples). This makes it only applicable to rather few (a couple of
 * thousand) training examples. In case a linear kernel is used RR is closely
 * related to Fishers Linear Discriminant (cf. LDA).
 *
 * Internally (for linear kernels) it is solved via minimizing the following system
 *
 * \f[
 * \frac{1}{2}\left(\sum_{i=1}^N(y_i-{\bf w}\cdot {\bf x}_i)^2 + \tau||{\bf w}||^2\right)
 * \f]
 * 
 * which is boils down to solving a linear system
 *
 * \f[
 * {\bf w} = \left(\tau {\bf I}+ \sum_{i=1}^N{\bf x}_i{\bf x}_i^T\right)^{-1}\left(\sum_{i=1}^N y_i{\bf x}_i\right)
 * \f]
 *
 * and in the kernel case
 * \f[
 * {\bf \alpha}=\left({\bf K}+\tau{\bf I}\right)^{-1}{\bf y}
 * \f]
 * where K is the kernel matrix and y the vector of labels. The expressed
 * solution can again be written as a linear combination of kernels (cf.
 * CKernelMachine) with bias \f$b=0\f$.
 */
class CKRR : public CKernelMachine
{
	public:
		/** default constructor */
		CKRR();

		/** constructor
		 *
		 * @param tau regularization constant tau
		 * @param k kernel
		 * @param lab labels
		 */
		CKRR(float64_t tau, CKernel* k, CLabels* lab);
		virtual ~CKRR();

		/** set regularization constant
		 *
		 * @param t new tau
		 */
		inline void set_tau(float64_t t) { tau = t; };

		/** train regression
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based regressors are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/** classify regression
		 *
		 * @return resulting labels
		 */
		virtual CLabels* classify();

		/** classify one example
		 *
		 * @param num which example to classify
		 * @return result
		 */
		virtual float64_t classify_example(int32_t num);

		/** load regression from file
		 *
		 * @param srcfile file to load from
		 * @return if loading was successful
		 */
		virtual bool load(FILE* srcfile);

		/** save regression to file
		 *
		 * @param dstfile file to save to
		 * @return if saving was successful
		 */
		virtual bool save(FILE* dstfile);

		/** get classifier type
		 *
		 * @return classifier type KRR
		 */
		inline virtual EClassifierType get_classifier_type()
		{
			return CT_KRR;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "KRR"; }

	private:
		/** alpha */
		float64_t *alpha;
		/** regularization parameter tau */
		float64_t tau;
};
}
#endif // HAVE_LAPACK
#endif // _KRR_H__
