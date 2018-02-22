/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those
 * of the authors and should not be interpreted as representing official
 * policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/evaluation/Calibration.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/lib/config.h>
#include <shogun/machine/Machine.h>

#ifndef _CROSS_VALIDATED_CALIBRATION_H__
#define _CROSS_VALIDATED_CALIBRATION_H__

namespace shogun
{

	class CCrossValidatedCalibration : public CMachine
	{

	public:
		CCrossValidatedCalibration();

		/** constructor
		 * @param machine learning machine to use
		 * @param labels labels that correspond to the features
		 * @param splitting_strategy splitting strategy to use
		 * @param calibrator calibration machine to use
		 * @param autolock autolock
		 */
		CCrossValidatedCalibration(
		    CMachine* machine, CLabels* labels,
		    CSplittingStrategy* splitting_strategy,
		    CCalibrationMethod* calibration_method);

		virtual ~CCrossValidatedCalibration();

		virtual const char* get_name() const
		{
			return "CrossValidatedCalibration";
		}

		virtual EProblemType get_machine_problem_type() const;

		virtual bool train(CFeatures* data = NULL);

		virtual CBinaryLabels* apply_binary(CFeatures* features = NULL);

		virtual CBinaryLabels*
		apply_locked_binary(SGVector<index_t> subset_indices);

		virtual bool train_locked(SGVector<index_t> indices);

		virtual CMulticlassLabels*
		apply_locked_multiclass(SGVector<index_t> subset_indices);

		virtual CMulticlassLabels* apply_multiclass(CFeatures* features);

		/** get learning machine
		*/
		CMachine* get_machine() const;

	private:
		void init();

		template <typename T>
		CMulticlassLabels* get_multiclass_result(T training_data);

		template <typename T>
		CBinaryLabels* get_binary_result(T training_data);

		CLabels* apply_once(CMachine* machine, CFeatures* features);

		CLabels*
		apply_once(CMachine* machine, SGVector<index_t> subset_indices);

	private:
		CDynamicObjectArray* m_calibration_machines;
		CMachine* m_machine;
		CLabels* m_labels;
		CSplittingStrategy* m_splitting_strategy;
		CCalibrationMethod* m_calibration_method;
	};
}
#endif