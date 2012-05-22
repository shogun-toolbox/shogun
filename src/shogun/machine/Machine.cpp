/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/machine/Machine.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/ParameterMap.h>

using namespace shogun;

CMachine::CMachine() : CSGObject(), m_max_train_time(0), m_labels(NULL),
		m_solver_type(ST_AUTO)
{
	m_data_locked=false;
	m_store_model_features=false;

	SG_ADD(&m_max_train_time, "max_train_time",
	       "Maximum training time.", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*) &m_solver_type, "solver_type",
	       "Type of solver.", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &m_labels, "labels",
	       "Labels to be used.", MS_NOT_AVAILABLE);
	SG_ADD(&m_store_model_features, "store_model_features",
	       "Should feature data of model be stored after training?", MS_NOT_AVAILABLE);
	SG_ADD(&m_data_locked, "data_locked",
			"Indicates whether data is locked", MS_NOT_AVAILABLE);

	m_parameter_map->put(
		new SGParamInfo("data_locked", CT_SCALAR, ST_NONE, PT_BOOL, 1),
		new SGParamInfo()
	);

	m_parameter_map->finalize_map();
}

CMachine::~CMachine()
{
	SG_UNREF(m_labels);
}

bool CMachine::train(CFeatures* data)
{
	/* not allowed to train on locked data */
	if (m_data_locked)
	{
		SG_ERROR("%s::train data_lock() was called, only train_locked() is"
				" possible. Call data_unlock if you want to call train()\n",
				get_name());
	}

    if (m_labels == NULL)
        SG_ERROR("%s@%p: No labels given", get_name(), this);
    m_labels->ensure_valid(get_name());

	bool result = train_machine(data);

	if (m_store_model_features)
		store_model_features();

	return result;
}

void CMachine::set_labels(CLabels* lab)
{
    if (lab != NULL)
        if (!is_label_valid(lab))
            SG_ERROR("Invalid label for %s", get_name());

	SG_UNREF(m_labels);
	SG_REF(lab);
	m_labels = lab;
}

CLabels* CMachine::get_labels()
{
	SG_REF(m_labels);
	return m_labels;
}

void CMachine::set_max_train_time(float64_t t)
{
	m_max_train_time = t;
}

float64_t CMachine::get_max_train_time()
{
	return m_max_train_time;
}

EMachineType CMachine::get_classifier_type()
{
	return CT_NONE;
}

void CMachine::set_solver_type(ESolverType st)
{
	m_solver_type = st;
}

ESolverType CMachine::get_solver_type()
{
	return m_solver_type;
}

void CMachine::set_store_model_features(bool store_model)
{
	m_store_model_features = store_model;
}

void CMachine::data_lock(CLabels* labs, CFeatures* features)
{
	SG_DEBUG("entering %s::data_lock\n", get_name());
	if (!supports_locking())
	{
		{
			SG_ERROR("%s::data_lock(): Machine does not support data locking!\n",
					get_name());
		}
	}

	if (!labs)
	{
		SG_ERROR("%s::data_lock() is not possible will NULL labels!\n",
				get_name());
	}

	/* first set labels */
	set_labels(labs);

	if (m_data_locked)
	{
		SG_ERROR("%s::data_lock() was already called. Dont lock twice!",
				get_name());
	}

	m_data_locked=true;
	SG_DEBUG("leaving %s::data_lock\n", get_name());
}

void CMachine::data_unlock()
{
	SG_DEBUG("entering %s::data_lock\n", get_name());
	if (m_data_locked)
		m_data_locked=false;

	SG_DEBUG("leaving %s::data_lock\n", get_name());
}

CLabels* CMachine::apply(CFeatures* data)
{
	switch (get_machine_problem_type())
	{
		case PT_BINARY:
			return apply_binary(data);
		case PT_REGRESSION:
			return apply_regression(data);
		case PT_MULTICLASS:
			return apply_multiclass(data);
		default: SG_ERROR("Unknown problem type");
	}
	return NULL;
}

CBinaryLabels* CMachine::apply_binary(CFeatures* data)
{
	SG_ERROR("This machine does not support apply_binary()\n");
	return NULL;
}

CRegressionLabels* CMachine::apply_regression(CFeatures* data)
{
	SG_ERROR("This machine does not support apply_regression()\n");
	return NULL;
}

CMulticlassLabels* CMachine::apply_multiclass(CFeatures* data)
{
	SG_ERROR("This machine does not support apply_multiclass()\n");
	return NULL;
}


