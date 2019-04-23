/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Fernando Iglesias, 
 *          Evan Shelhamer
 */

#ifndef _LEASTSQUARESREGRESSION_H__
#define _LEASTSQUARESREGRESSION_H__

#include <shogun/lib/config.h>
#include <shogun/regression/Regression.h>
#include <shogun/regression/LinearRidgeRegression.h>

#ifdef HAVE_LAPACK

#include <shogun/machine/LinearMachine.h>

namespace shogun
{
	/** @brief class to perform Least Squares Regression
	 *
	 * Same as CLinearRidgeRegression, but without a regularization term.
	 */
	class LeastSquaresRegression : public LinearRidgeRegression
	{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);

		/** default constructor */
		LeastSquaresRegression();

		/** constructor
		 *
		 * @param data training data
		 * @param lab labels
		 */
		LeastSquaresRegression(std::shared_ptr<DenseFeatures<float64_t>> data, std::shared_ptr<Labels> lab);
		virtual ~LeastSquaresRegression() {}

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
