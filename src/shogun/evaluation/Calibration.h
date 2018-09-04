/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Dhruv Arya
 */

#ifndef _CALIBRATION_H__
#define _CALIBRATION_H__

#include <shogun/lib/config.h>

#include <shogun/machine/Machine.h>

namespace shogun
{
	/** @brief Base class for all calibration methods. Call fit_binary() or
	fit_multiclass() to
	fit the parameters on the predictions and the true labels. Call calibrate to
	calibrate predictions. **/
	class CCalibration : public CSGObject
	{
	public:
		/** Constructor. */
		CCalibration()
		{
		}
		/** Destructor. */
		virtual ~CCalibration()
		{
		}

		virtual const char* get_name() const
		{
			return "Calibration";
		}

		/** Fit calibration parameters for binary labels.
		* @param predictions The predictions outputted by the machine
		* @param targets The true labels corresponding to the predictions
		* @return Indicates whether the calibration was succesful
		**/
		virtual bool
		fit_binary(CBinaryLabels* predictions, CBinaryLabels* targets) = 0;

		/** Calibrate binary predictions based on parameters learned by calling
		*fit.
		* @param predictions The predictions outputted by the machine
		* @return Calibrated binary labels
		**/
		virtual CBinaryLabels* calibrate_binary(CBinaryLabels* predictions) = 0;

		/** Fit calibration parameters for multiclass labels.
		* @param predictions The predictions outputted by the machine
		* @param targets The true labels corresponding to the predictions
		* @return Indicates whether the calibration was succesful
		**/
		virtual bool fit_multiclass(
		    CMulticlassLabels* predictions, CMulticlassLabels* targets) = 0;

		/** Calibrate multiclass predictions based on parameters learned by
		*calling fit.
		* @param predictions The predictions outputted by the machine
		* @return Calibrated binary labels
		**/
		virtual CMulticlassLabels*
		calibrate_multiclass(CMulticlassLabels* predictions) = 0;
	};
}
#endif