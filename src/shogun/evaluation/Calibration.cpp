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

#include <shogun/lib/config.h>

#include <shogun/evaluation/Calibration.h>
#include <shogun/evaluation/CalibrationMethod.h>
#include <shogun/labels/LabelsFactory.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

CCalibration::CCalibration() : CMachine()
{
	init();
}

CCalibration::~CCalibration()
{
	SG_UNREF(m_calibration_machines)
	SG_UNREF(m_machine)
	SG_UNREF(m_labels)
	SG_UNREF(m_method)
}

void CCalibration::init()
{
	m_machine = new CMachine();
	m_labels = new CBinaryLabels();
	m_method = new CCalibrationMethod();
	m_calibration_machines = new CDynamicObjectArray();

	SG_ADD(
	    (CSGObject**)&m_machine, "m_machine", "learning machine to use",
	    MS_NOT_AVAILABLE);
	SG_ADD(
	    (CSGObject**)&m_labels, "m_labels", "target_labels", MS_NOT_AVAILABLE);
	SG_ADD(
	    (CSGObject**)&m_method, "m_method", "calibration method",
	    MS_NOT_AVAILABLE);
	SG_ADD(
	    (CSGObject**)&m_calibration_machines, "m_calibration_machines",
	    "array of calibration method machines", MS_NOT_AVAILABLE);
}

template <typename T>
CBinaryLabels* CCalibration::get_binary_result(T data)
{
	CLabels* result = apply_once(data);
	CBinaryLabels* result_labels = CLabelsFactory::to_binary(result);
	CCalibrationMethod* method =
	    (CCalibrationMethod*)m_calibration_machines->get_element(0);
	SGVector<float64_t> confidence_values =
	    method->apply_binary(result_labels->get_values());
	result_labels->set_values(confidence_values);

	return result_labels;
}

CBinaryLabels* CCalibration::apply_binary(CFeatures* features)
{
	return get_binary_result(features);
}

CBinaryLabels*
CCalibration::apply_locked_binary(SGVector<index_t> subset_indices)
{
	return get_binary_result(subset_indices);
}

CMulticlassLabels* CCalibration::get_multiclass_result(
    CMulticlassLabels* result_labels, index_t num_calibration_machines)
{
	for (index_t i = 0; i < num_calibration_machines; ++i)
	{
		CCalibrationMethod* method =
		    (CCalibrationMethod*)m_calibration_machines->get_element(i);
		SGVector<float64_t> confidence_values =
		    method->apply_binary(result_labels->get_multiclass_confidences(i));
		result_labels->set_multiclass_confidences(i, confidence_values);
		SG_UNREF(method)
	}

	SGVector<float64_t> temp_confidences =
	    result_labels->get_multiclass_confidences(0);
	temp_confidences.zero();

	index_t num_classes = result_labels->get_num_classes();

	index_t num_samples = temp_confidences.vlen;

// normalize the probabilities
#pragma omp parallel for
	for (index_t i = 0; i < num_classes; ++i)
	{
		SGVector<float64_t> confidence_values =
		    result_labels->get_multiclass_confidences(i);
		linalg::add(temp_confidences, confidence_values, temp_confidences, 1.0);
	}
#pragma omp parallel for
	for (index_t i = 0; i < num_classes; ++i)
	{
		SGVector<float64_t> confidence_values =
		    result_labels->get_multiclass_confidences(i);
		for (index_t j = 0; j < num_samples; ++j)
		{
			confidence_values[j] /= temp_confidences[j];
		}
		result_labels->set_multiclass_confidences(i, confidence_values);
	}

	return result_labels;
}

CMulticlassLabels* CCalibration::apply_multiclass(CFeatures* features)
{
	index_t num_calibration_machines =
	    m_calibration_machines->get_num_elements();
	CLabels* result = m_machine->apply(features);
	CMulticlassLabels* result_labels = CLabelsFactory::to_multiclass(result);
	return get_multiclass_result(result_labels, num_calibration_machines);
}

CMulticlassLabels*
CCalibration::apply_locked_multiclass(SGVector<index_t> subset_indices)
{
	index_t num_calibration_machines =
	    m_calibration_machines->get_num_elements();
	CLabels* result = m_machine->apply_locked(subset_indices);
	CMulticlassLabels* result_labels = CLabelsFactory::to_multiclass(result);
	return get_multiclass_result(result_labels, num_calibration_machines);
}

EProblemType CCalibration::get_machine_problem_type() const
{
	return m_machine->get_machine_problem_type();
}

bool CCalibration::train_one_machine(CFeatures* features)
{
	return m_machine->train(features);
}

bool CCalibration::train_one_machine(SGVector<index_t> subset_indices)
{
	return m_machine->train_locked(subset_indices);
}

CLabels* CCalibration::apply_once(CFeatures* features)
{
	return m_machine->apply(features);
}

CLabels* CCalibration::apply_once(SGVector<index_t> subset_indices)
{
	return m_machine->apply_locked(subset_indices);
}

template <typename T>
bool CCalibration::train_calibration_machine(T training_data)
{
	CCalibrationMethod* calibration_machine = NULL;
	if (get_machine_problem_type() == PT_MULTICLASS)
	{
		SGVector<float64_t> confidences;
		index_t num_calibration_machines =
		    (CLabelsFactory::to_multiclass(get_labels()))->get_num_classes();
		m_calibration_machines =
		    new CDynamicObjectArray(num_calibration_machines);
		train_one_machine(training_data);
		CLabels* result = apply_once(training_data);
		CMulticlassLabels* result_labels =
		    CLabelsFactory::to_multiclass(result);

		for (index_t i = 0; i < num_calibration_machines; ++i)
		{
			confidences = result_labels->get_multiclass_confidences(i);

			calibration_machine = (CCalibrationMethod*)m_method->clone();
			if (!calibration_machine->train(confidences))
			{
				return false;
			}
			m_calibration_machines->set_element(calibration_machine, i);
			SG_UNREF(calibration_machine)
		}
		SG_UNREF(result_labels)
	}
	else
	{
		SGVector<float64_t> confidences;
		m_calibration_machines = new CDynamicObjectArray(1);
		train_one_machine(training_data);
		CLabels* result = apply_once(training_data);
		CBinaryLabels* result_labels = CLabelsFactory::to_binary(result);

		confidences = result_labels->get_values();

		SG_UNREF(result_labels)

		calibration_machine = (CCalibrationMethod*)m_method->clone();
		if (!calibration_machine->train(confidences))
		{
			return false;
		}
		m_calibration_machines->set_element(calibration_machine, 0);
	}

	return true;
}

bool CCalibration::train(CFeatures* features)
{
	return train_calibration_machine(features);
}

bool CCalibration::train_locked(SGVector<index_t> subset_indices)
{
	return train_calibration_machine(subset_indices);
}

void CCalibration::set_calibration_method(CCalibrationMethod* method)
{
	SG_UNREF(m_method);
	m_method = method;
	SG_REF(m_method);
}

void CCalibration::set_machine(CMachine* machine)
{
	SG_UNREF(m_machine);
	m_machine = machine;
	SG_REF(m_machine);
}

CMachine* CCalibration::get_machine()
{
	SG_REF(m_machine);
	return m_machine;
}

CCalibrationMethod* CCalibration::get_calibration_method()
{
	SG_REF(m_method);
	return m_method;
}