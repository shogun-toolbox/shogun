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
#include <shogun/evaluation/CrossValidatedCalibration.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/labels/LabelsFactory.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;
using namespace linalg;

CCrossValidatedCalibration::CCrossValidatedCalibration() : CMachine()
{
	init();
}

CCrossValidatedCalibration::CCrossValidatedCalibration(
    CMachine* machine, CLabels* labels, CSplittingStrategy* splitting_strategy,
    CCalibrationMethod* calibration_method)
    : CMachine()
{
	init();

	m_machine = machine;
	m_labels = labels;
	m_splitting_strategy = splitting_strategy;
	m_calibration_method = calibration_method;

	SG_REF(m_machine)
	SG_REF(m_labels)
	SG_REF(m_splitting_strategy)
	SG_REF(m_calibration_method)
}

CCrossValidatedCalibration::~CCrossValidatedCalibration()
{
	SG_UNREF(m_machine);
	SG_UNREF(m_labels);
	SG_UNREF(m_splitting_strategy);
	SG_UNREF(m_calibration_method);
	SG_UNREF(m_calibration_machines);
}

void CCrossValidatedCalibration::init()
{
	m_machine = new CMachine();
	m_labels = new CBinaryLabels();
	m_splitting_strategy = new CStratifiedCrossValidationSplitting();
	m_calibration_method = new CCalibrationMethod();
	m_calibration_machines = new CDynamicObjectArray();

	SG_ADD(
	    (CSGObject**)&m_machine, "m_machine", "machine to be calibrated",
	    MS_NOT_AVAILABLE);
	SG_ADD(
	    (CSGObject**)&m_labels, "m_labels", "target labels", MS_NOT_AVAILABLE);
	SG_ADD(
	    (CSGObject**)&m_splitting_strategy, "m_splitting_strategy",
	    "splitting strategy for cross-validation", MS_NOT_AVAILABLE);
	SG_ADD(
	    (CSGObject**)&m_calibration_method, "m_calibration_method",
	    "method for calibrating the predictions", MS_NOT_AVAILABLE);
	SG_ADD(
	    (CSGObject**)&m_calibration_machines, "m_calibration_machines",
	    "array of calibration machines", MS_NOT_AVAILABLE);
}

EProblemType CCrossValidatedCalibration::get_machine_problem_type() const
{
	return m_machine->get_machine_problem_type();
}

bool CCrossValidatedCalibration::train(CFeatures* data)
{
	// code borrowed from Calibration.cpp
	index_t num_subsets = m_splitting_strategy->get_num_subsets();

	m_calibration_machines = new CDynamicObjectArray(num_subsets);

	SG_DEBUG(
	    "building index sets for %d-fold cross-validated calibration\n",
	    num_subsets)

	/* build index sets */
	m_splitting_strategy->build_subsets();

	if (m_machine->is_data_locked())
	{
		SG_ERROR("cannot run train on locked data use train_locked instead\n")
		return false;
	}
	SG_DEBUG("starting unlocked calibration\n", get_name())
	/* tell machine to store model internally
	 * (otherwise changing subset of features will kaboom the classifier) */
	m_machine->set_store_model_features(true);

	for (index_t i = 0; i < num_subsets; ++i)
	{
		CMachine* machine;
		CFeatures* features;
		CLabels* labels;

		if (get_global_parallel()->get_num_threads() == 1)
		{
			machine = m_machine;
			features = data;
		}
		else
		{
			machine = (CMachine*)m_machine->clone();
			features = (CFeatures*)data->clone();
		}

		/* set feature subset for training */
		SGVector<index_t> inverse_subset_indices =
		    m_splitting_strategy->generate_subset_inverse(i);

		features->add_subset(inverse_subset_indices);

		/* set label subset for training */
		if (get_global_parallel()->get_num_threads() == 1)
			labels = m_labels;
		else
			labels = machine->get_labels();
		labels->add_subset(inverse_subset_indices);

		SG_DEBUG("training set %d:\n", i)
		if (io->get_loglevel() == MSG_DEBUG)
		{
			SGVector<index_t>::display_vector(
			    inverse_subset_indices.vector, inverse_subset_indices.vlen,
			    "training indices");
		}

		/* train machine on training features and remove subset */
		SG_DEBUG("starting training\n")
		machine->train(features);
		SG_DEBUG("finished training\n")
		features->remove_subset();
		labels->remove_subset();

		SGVector<index_t> subset_indices =
		    m_splitting_strategy->generate_subset_indices(i);
		features->add_subset(subset_indices);

		/* set label subset for calibration */
		labels->add_subset(subset_indices);

		SG_DEBUG("test set %d:\n", i)
		if (io->get_loglevel() == MSG_DEBUG)
		{
			SGVector<index_t>::display_vector(
			    subset_indices.vector, subset_indices.vlen, "test indices");
		}

		CCalibration* calibrator = new CCalibration();
		calibrator->set_machine((CMachine*)machine->clone());
		calibrator->set_labels((CLabels*)labels->clone());
		calibrator->set_calibration_method(
		    (CCalibrationMethod*)m_calibration_method->clone());
		bool trained = calibrator->train((CFeatures*)features->clone());

		if (!trained)
		{
			return false;
		}

		m_calibration_machines->set_element(calibrator, i);

		SG_DEBUG("finished calibration\n")
		features->remove_subset();

		/* clean up, remove subsets */
		labels->remove_subset();
		if (get_global_parallel()->get_num_threads() != 1)
		{
			SG_UNREF(machine);
			SG_UNREF(features);
			SG_UNREF(labels);
		}
	}

	SG_DEBUG("done unlocked calibration\n", get_name())
	return true;
}

bool CCrossValidatedCalibration::train_locked(SGVector<index_t> indices)
{

	index_t num_subsets = m_splitting_strategy->get_num_subsets();

	m_calibration_machines = new CDynamicObjectArray(num_subsets);

	SG_DEBUG(
	    "building index sets for %d-fold cross-validated calibration\n",
	    num_subsets)

	/* build index sets */
	m_splitting_strategy->build_subsets();

	if (!m_machine->is_data_locked())
	{
		SG_ERROR("please lock the data before running train_locked")
		return false;
	}

	SG_DEBUG("starting unlocked calibration\n", get_name())
	/* tell machine to store model internally
	 * (otherwise changing subset of features will kaboom the classifier) */
	m_machine->set_store_model_features(true);

	for (index_t i = 0; i < num_subsets; ++i)
	{
		/* index subset for training, will be freed below */
		SGVector<index_t> inverse_subset_indices =
		    m_splitting_strategy->generate_subset_inverse(i);

		/* train machine on training features */
		m_machine->train_locked(inverse_subset_indices);

		/* feature subset for calibration */
		SGVector<index_t> subset_indices =
		    m_splitting_strategy->generate_subset_indices(i);

		/* set subset for testing labels */
		m_labels->add_subset(subset_indices);

		/* produce output for desired indices */
		CCalibration* calibrator = new CCalibration();
		calibrator->set_machine((CMachine*)m_machine->clone());
		calibrator->set_labels((CLabels*)m_labels->clone());
		calibrator->set_calibration_method(
		    (CCalibrationMethod*)m_calibration_method->clone());

		bool trained = calibrator->train_locked(subset_indices);

		if (!trained)
		{
			return false;
		}

		m_calibration_machines->set_element(calibrator, i);

		/* remove subset to prevent side effects */
		m_labels->remove_subset();

		SG_DEBUG("done locked calibration\n", get_name())
	}
	return true;
}

CLabels* CCrossValidatedCalibration::apply_once(
    CMachine* machine, SGVector<index_t> subset_indices)
{
	return machine->apply_locked(subset_indices);
}

CLabels*
CCrossValidatedCalibration::apply_once(CMachine* machine, CFeatures* features)
{
	return machine->apply(features);
}

template <typename T>
CBinaryLabels* CCrossValidatedCalibration::get_binary_result(T data)
{
	index_t num_machines = m_calibration_machines->get_num_elements();

	CBinaryLabels* temp_result;
	CMachine* temp_machine;
	SGVector<float64_t> result_values;
	CBinaryLabels* result_labels;
	CLabels* result;

	temp_machine = (CMachine*)m_calibration_machines->get_element(0);
	result = apply_once(temp_machine, data);
	result_labels = CLabelsFactory::to_binary(result);
	result_values = result_labels->get_values();

	for (index_t i = 0; i < num_machines; ++i)
	{
		temp_machine = (CMachine*)m_calibration_machines->get_element(0);
		result = apply_once(temp_machine, data);
		temp_result = CLabelsFactory::to_binary(result);
		result_values += temp_result->get_values();
	}

	linalg::scale(result_values, result_values, 1.0 / num_machines);
	result_labels->set_values(result_values);

	return result_labels;
}

CBinaryLabels* CCrossValidatedCalibration::apply_binary(CFeatures* features)
{
	return get_binary_result(features);
}

CBinaryLabels* CCrossValidatedCalibration::apply_locked_binary(
    SGVector<index_t> subset_indices)
{
	return get_binary_result(subset_indices);
}

template <typename T>
CMulticlassLabels* CCrossValidatedCalibration::get_multiclass_result(T data)
{
	index_t num_machines = m_calibration_machines->get_num_elements();

	CMulticlassLabels* labels = CLabelsFactory::to_multiclass(m_labels);
	index_t num_classes = labels->get_num_classes();

	CMulticlassLabels* temp_result;
	CMachine* temp_machine;
	SGVector<float64_t> result_values, temp_values;
	CMulticlassLabels* result_labels;
	CLabels* result;

	temp_machine = (CMachine*)m_calibration_machines->get_element(0);
	result = apply_once(temp_machine, data);
	result_labels = CLabelsFactory::to_multiclass(result);

	for (index_t i = 1; i < num_machines; ++i)
	{
		temp_machine = (CMachine*)m_calibration_machines->get_element(i);
		result = apply_once(temp_machine, data);
		temp_result = CLabelsFactory::to_multiclass(result);
		for (index_t j = 0; j < num_classes; ++j)
		{
			result_values = temp_result->get_multiclass_confidences(j);
			temp_values = result_labels->get_multiclass_confidences(j);
			temp_values += result_values;
			result_labels->set_multiclass_confidences(j, temp_values);
		}
		SG_UNREF(temp_machine)
	}

	for (index_t i = 0; i < num_classes; ++i)
	{
		temp_values = result_labels->get_multiclass_confidences(i);

		linalg::scale(temp_values, temp_values, 1.0 / num_machines);
		result_labels->set_multiclass_confidences(i, temp_values);
	}

	return result_labels;
}

CMulticlassLabels*
CCrossValidatedCalibration::apply_multiclass(CFeatures* features)
{
	return get_multiclass_result(features);
}

CMulticlassLabels* CCrossValidatedCalibration::apply_locked_multiclass(
    SGVector<index_t> subset_indices)
{
	return get_multiclass_result(subset_indices);
}
