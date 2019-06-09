/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Fernando Iglesias, 
 *          Evan Shelhamer
 */

#ifndef _LEASTSQUARESREGRESSION_H__
#define _LEASTSQUARESREGRESSION_H__

#include <shogun/regression/LinearRidgeRegression.h>

namespace shogun
{
	/** @brief class to perform Least Squares Regression
	 *
	 * Same as CLinearRidgeRegression, but without a regularization term.
	 */
	class CLeastSquaresRegression : public CLinearRidgeRegression
	{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);

		/** default constructor */
		CLeastSquaresRegression();

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
#endif // _LEASTSQUARESREGRESSION_H__
