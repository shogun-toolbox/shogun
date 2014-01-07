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

#ifndef _KERNELRIDGEREGRESSION_H__
#define _KERNELRIDGEREGRESSION_H__

#include <lib/config.h>
#include <regression/Regression.h>

#ifdef HAVE_LAPACK

#include <machine/KernelMachine.h>

namespace shogun
{

/** which training method to use for KRR */
enum ETrainingType
{
	/// via pseudo inverse
	PINV=1,
	/// or gauss-seidel iterative method
	GS=2
};

/** @brief Class KernelRidgeRegression implements Kernel Ridge Regression - a regularized least square
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
 * which boils down to solving a linear system
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
class CKernelRidgeRegression : public CKernelMachine
{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);

		/** default constructor */
		CKernelRidgeRegression();

		/** constructor
		 *
		 * @param tau regularization constant tau
		 * @param k kernel
		 * @param lab labels
		 * @param m method to use for training PINV (pseudo inverse by default)
		 */
		CKernelRidgeRegression(float64_t tau, CKernel* k, CLabels* lab, ETrainingType m=PINV);

		/** default destructor */
		virtual ~CKernelRidgeRegression() {}

		/** set regularization constant
		 *
		 * @param tau new tau
		 */
		inline void set_tau(float64_t tau) { m_tau = tau; };

		/** set convergence precision for gauss seidel method
		 *
		 * @param epsilon new epsilon
		 */
		inline void set_epsilon(float64_t epsilon) { m_epsilon = epsilon; }

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
		 * @return classifier type KernelRidgeRegression
		 */
		virtual EMachineType get_classifier_type()
		{
			return CT_KERNELRIDGEREGRESSION;
		}

		/** @return object name */
		virtual const char* get_name() const { return "KernelRidgeRegression"; }

	protected:
		/** train regression
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based regressors are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

	private:
		void init();

		/** train regression using Gauss-Seidel iterative method
		 *
		 * @return whether training was successful
		 */
		bool train_machine_gs();

		/** train regression using pinv
		 *
		 * @return whether training was successful
		 */
		bool train_machine_pinv();

	private:
		/** regularization parameter tau */
		float64_t m_tau;

		/** epsilon constant */
		float64_t m_epsilon;

		/** training function */
		ETrainingType m_train_func;
};
}

#endif // HAVE_LAPACK
#endif // _KERNELRIDGEREGRESSION_H__
