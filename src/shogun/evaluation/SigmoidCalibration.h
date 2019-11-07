/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Dhruv Arya
 */

#ifndef _SIGMOID_CALIBRATION_H__
#define _SIGMOID_CALIBRATION_H__

#include <shogun/lib/config.h>

#include <shogun/evaluation/Calibration.h>
#include <shogun/mathematics/Statistics.h>

namespace shogun
{
	/** @brief Calibrates labels based on Platt Scaling [1]. Note that first
	* calibration parameters need to be fitted by
	* calling fit_binary() or fit_multiclass(). Then call calibrate on
	* calibrate_binary() or calibrate_multiclass() on
	* labels to calibrate them.
	* Usually this done using the training data and labels.
	* [1] Platt J. Probabilistic outputs for support vector machines and
	* comparisons to regularized likelihood methods.
	* Advances in large margin classifiers. 1999
	*/
	class SigmoidCalibration : public Calibration
	{
	public:
		/** Constructor. */
		SigmoidCalibration();

		/** Destructor. */
		virtual ~SigmoidCalibration();

		/** Get name. */
		virtual const char* get_name() const
		{
			return "SigmoidCalibration";
		}

		/** Fit sigmoid parameters for binary labels.
		* @param predictions The predictions outputted by the machine
		* @param targets The true labels corresponding to the predictions
		* @return Indicates whether the calibration was succesful
		**/
		virtual bool
		fit_binary(std::shared_ptr<BinaryLabels> predictions, std::shared_ptr<BinaryLabels> targets);

		/** Calibrate binary predictions based on parameters learned by calling
		*fit.
		* @param predictions The predictions outputted by the machine
		* @return Calibrated binary labels
		**/
		virtual std::shared_ptr<BinaryLabels> calibrate_binary(std::shared_ptr<BinaryLabels> predictions);

		/** Fit calibration parameters for multiclass labels. Fits sigmoid
		* parameters for each class seperately.
		* @param predictions The predictions outputted by the machine
		* @param targets The true labels corresponding to the predictions
		* @return Indicates whether the calibration was succesful
		**/
		virtual bool fit_multiclass(
		    std::shared_ptr<MulticlassLabels> predictions, std::shared_ptr<MulticlassLabels> targets);

		/** Calibrate multiclass predictions based on parameters learned by
		*calling fit.
		* The predictions are normalized over all classes.
		* @param predictions The predictions outputted by the machine
		* @return Calibrated binary labels
		**/
		virtual std::shared_ptr<MulticlassLabels>
		calibrate_multiclass(std::shared_ptr<MulticlassLabels> predictions);

		/** Set maximum number of iterations
		* @param maxiter maximum number of iterations
		*/
		void set_maxiter(index_t maxiter);

		/** Get max iterations
		* @return maximum number of iterations
		*/
		index_t get_maxiter();

		/** Set min step
		* @param minstep min step taken in line search
		*/
		void set_minstep(float64_t minstep);

		/** Get min step
		* @return minimum steps taken in line search
		*/
		float64_t get_minstep();

		/** Set sigma
		* @param sigma Set to a value greater than 0 to ensure that the Hessian
		* matrix is positive semi-definite
		*/
		void set_sigma(float64_t sigma);

		/** Get sigma
		* @return sigma
		*/
		float64_t get_sigma();

		/** Get epsilon
		* @param epsilon stopping criteria
		*/
		void set_epsilon(float64_t epsilon);

		/** Get epsilon
		* @return stopping critera
		*/
		float64_t get_epsilon();

	private:
		/** Initialize parameters */
		void init();

		/** Helper function that calibrates values of given vector using the
		* given sigmoid parameters
		* @param values The values to be calibrated
		* @param params The sigmoid paramters to be used for calibration
		*/
		SGVector<float64_t> calibrate_values(
		    SGVector<float64_t> values, Statistics::SigmoidParamters params);

	private:
		/** Stores parameter A of sigmoid for each class. In case of binary
		 * labels, only one pair of parameters are stored. */
		SGVector<float64_t> m_sigmoid_as;
		/** Stores parameter B of sigmoid for each class. */
		SGVector<float64_t> m_sigmoid_bs;
		/** Maximum number of iterations. */
		index_t m_maxiter;
		/** Minimum step taken in line search. */
		float64_t m_minstep;
		/** Positive number to ensure positive semi-definite Hessian matrix */
		float64_t m_sigma;
		/** Stopping criteria of search */
		float64_t m_epsilon;
	};
}
#endif