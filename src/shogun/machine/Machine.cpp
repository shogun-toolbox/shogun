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

	bool result = train_machine(data);

	if (m_store_model_features)
		store_model_features();

	return result;
}

float64_t CMachine::apply(int32_t num)
{
	SG_NOTIMPLEMENTED;
	return CMath::INFTY;
}

bool CMachine::load(FILE* srcfile)
{
	ASSERT(srcfile);
	return false;
}

bool CMachine::save(FILE* dstfile)
{
	ASSERT(dstfile);
	return false;
}

void CMachine::set_labels(CLabels* lab)
{
	SG_UNREF(m_labels);
	SG_REF(lab);
	m_labels = lab;
}

CLabels* CMachine::get_labels()
{
	SG_REF(m_labels);
	return m_labels;
}

float64_t CMachine::get_label(int32_t i)
{
	if (!m_labels)
		SG_ERROR("No Labels assigned\n");
	
	return m_labels->get_label(i);
}

void CMachine::set_max_train_time(float64_t t)
{
	m_max_train_time = t;
}

float64_t CMachine::get_max_train_time()
{
	return m_max_train_time;
}

EClassifierType CMachine::get_classifier_type()
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
	if (!labs)
	{
		SG_ERROR("%s::data_lock() is not possible will NULL labels!\n",
				get_name());
	}

	/* first set labels */
	set_labels(labs);

	/* if labels have a subset this might cause problems */
	if (m_labels->has_subset())
	{
		SG_ERROR("%s::data_lock() not possible if labels have a subset. Remove"
				" first!\n", get_name());
	}

	if (m_data_locked)
	{
		SG_ERROR("%s::data_lock() was already called. Dont lock twice!",
				get_name());
	}

	m_data_locked=true;
}

void CMachine::data_unlock()
{
	if (m_data_locked)
	{
		/* remove possible subset in labels */
		m_labels->remove_subset();
		m_data_locked=false;
	}
}
