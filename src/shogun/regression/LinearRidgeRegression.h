/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#ifndef _LINEARRIDGEREGRESSION_H__
#define _LINEARRIDGEREGRESSION_H__

#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK

#include <shogun/regression/Regression.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief Class LinearRidgeRegression implements Ridge Regression - a regularized least square
 * method for classification and regression.
 *
 * RR is closely related to Fishers Linear Discriminant (cf. LDA).
 *
 * Internally, it is solved via minimizing the following system
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
 * The expressed solution is a linear method with bias 0 (cf. CLinearMachine).
 */
class CLinearRidgeRegression : public CLinearMachine
{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);

		/** default constructor */
		CLinearRidgeRegression();

		/** constructor
		 *
		 * @param tau regularization constant tau
		 * @param data training data
		 * @param lab labels
		 */
		CLinearRidgeRegression(float64_t tau, CDenseFeatures<float64_t>* data, CLabels* lab);
		virtual ~CLinearRidgeRegression() {}

		/** set regularization constant
		 *
		 * @param tau new tau
		 */
		inline void set_tau(float64_t tau) { m_tau = tau; };

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
		 * @return classifier type LinearRidgeRegression
		 */
		virtual EMachineType get_classifier_type()
		{
			return CT_LINEARRIDGEREGRESSION;
		}

		/** @return object name */
		virtual const char* get_name() const { return "LinearRidgeRegression"; }

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

	protected:
		/** regularization parameter tau */
		float64_t m_tau;
};
}
#endif // HAVE_LAPACK
#endif // _LINEARRIDGEREGRESSION_H__
