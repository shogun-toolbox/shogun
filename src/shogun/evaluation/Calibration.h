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

#ifndef _CALIBRATION_H__
#define _CALIBRATION_H__

#include <shogun/lib/config.h>

#include <shogun/evaluation/CalibrationMethod.h>
#include <shogun/machine/Machine.h>

namespace shogun
{

	class CCalibration : public CMachine
	{
	public:
		/** constructor
		 */
		CCalibration();

		virtual ~CCalibration();

		virtual const char* get_name() const
		{
			return "Calibration";
		}

		virtual EProblemType get_machine_problem_type() const;

		virtual bool train(CFeatures* data = NULL);

		virtual bool train_locked(SGVector<index_t> subset_indices);

		virtual CBinaryLabels* apply_binary(CFeatures* features);

		virtual CMulticlassLabels* get_multiclass_result(
		    CMulticlassLabels* result_labels, index_t num_calibration_machines);

		virtual CMulticlassLabels* apply_multiclass(CFeatures* features);

		virtual CMulticlassLabels*
		apply_locked_multiclass(SGVector<index_t> subset_indices);

		virtual CBinaryLabels*
		apply_locked_binary(SGVector<index_t> subset_indices);

		virtual CMachine* get_machine();

		virtual void set_machine(CMachine* machine);

		virtual void
		set_calibration_method(CCalibrationMethod* calibration_method);

		virtual CCalibrationMethod* get_calibration_method();

	private:
		CLabels* apply_once(CFeatures* features);

		CLabels* apply_once(SGVector<index_t> subset_indices);

		template <typename T>
		bool train_calibration_machine(T training_data);

		template <typename T>
		CBinaryLabels* get_binary_result(T data);

		void init();

		bool train_one_machine(SGVector<index_t> subset_indices);

		bool train_one_machine(CFeatures* features);

	private:
		CMachine* m_machine;
		CDynamicObjectArray* m_calibration_machines;
		CCalibrationMethod* m_method;
	};
}
#endif