/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#ifndef _LEASTSQUARESREGRESSION_H__
#define _LEASTSQUARESREGRESSION_H__

#include <lib/config.h>
#include <regression/Regression.h>
#include <regression/LinearRidgeRegression.h>

#ifdef HAVE_LAPACK

#include <machine/LinearMachine.h>

namespace shogun
{
/** @brief class to perform Least Squares Regression
 *
 * Internally it is solved via minimizing the following system
 *
 * \f[
 * \frac{1}{2}\left(\sum_{i=1}^N(y_i-{\bf w}\cdot {\bf x}_i)^2\right)
 * \f]
 *
 * which boils down to solving the linear system
 *
 * \f[
 * {\bf w} = \left(\sum_{i=1}^N{\bf x}_i{\bf x}_i^T\right)^{-1}\left(\sum_{i=1}^N y_i{\bf x}_i\right)
 * \f]
 * where x are the training examples and y the vector of labels.
 *
 * The expressed solution is a linear method with bias 0 (cf. CLinearMachine).
 */
class CLeastSquaresRegression : public CLinearRidgeRegression
{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);

		/** default constructor */
		CLeastSquaresRegression();

		/** constructor
		 *
		 * @param data training data
		 * @param lab labels
		 */
		CLeastSquaresRegression(CDenseFeatures<float64_t>* data, CLabels* lab);
		virtual ~CLeastSquaresRegression() {}

		/** get classifier type
		 *
		 * @return classifier type LeastSquaresRegression
		 */
		virtual EMachineType get_classifier_type()
		{
			return CT_LEASTSQUARESREGRESSION;
		}

		/** @return object name */
		virtual const char* get_name() const { return "LeastSquaresRegression"; }

	private:
		void init();
};
}
#endif // HAVE_LAPACK
#endif // _LEASTSQUARESREGRESSION_H__
