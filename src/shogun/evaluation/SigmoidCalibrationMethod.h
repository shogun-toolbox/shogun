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

#ifndef _SIGMOID_CALIBRATION_METHOD_H__
#define _SIGMOID_CALIBRATION_METHOD_H__

#include <shogun/lib/config.h>

#include <shogun/evaluation/CalibrationMethod.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{

	class CSigmoidCalibrationMethod : public CCalibrationMethod
	{
	public:
		CSigmoidCalibrationMethod();

		CSigmoidCalibrationMethod(SGVector<float64_t> target_values);

		virtual ~CSigmoidCalibrationMethod();

		virtual const char* get_name() const
		{
			return "SigmoidCalibrationMethod";
		}

		virtual EProblemType get_machine_problem_type() const
		{
			return PT_BINARY;
		}

		virtual bool train(SGVector<float64_t> values);

		virtual SGVector<float64_t> apply_binary(SGVector<float64_t> values);

	private:
		void init();

	private:
		float64_t m_a, m_b;
	};
}
#endif