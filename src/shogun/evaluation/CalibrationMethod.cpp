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

#include <shogun/evaluation/CalibrationMethod.h>
#include <shogun/machine/Machine.h>

using namespace shogun;

CCalibrationMethod::CCalibrationMethod() : CMachine()
{
	init();
}

CCalibrationMethod::CCalibrationMethod(SGVector<float64_t> target_values)
{
	init();
	m_target_values = target_values;
}

CCalibrationMethod::~CCalibrationMethod()
{
}

void CCalibrationMethod::init()
{
	m_target_values = SGVector<float64_t>();
	SG_ADD(
	    &m_target_values, "m_target_values", "the true label values",
	    MS_NOT_AVAILABLE);
}

SGVector<float64_t> CCalibrationMethod::apply_binary(SGVector<float64_t> values)
{
	SG_NOTIMPLEMENTED
	return NULL;
}

bool CCalibrationMethod::train(SGVector<float64_t> values)
{
	SG_NOTIMPLEMENTED

	return true;
}

void CCalibrationMethod::set_target_values(SGVector<float64_t> target_values)
{
	m_target_values = target_values;
}