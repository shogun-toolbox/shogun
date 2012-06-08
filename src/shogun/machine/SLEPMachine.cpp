/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/machine/SLEPMachine.h>
#include <shogun/lib/slep/slep_options.h>

namespace shogun
{

CSLEPMachine::CSLEPMachine() :
	CLinearMachine(), m_z(1.0)
{
}

CSLEPMachine::CSLEPMachine(
     float64_t z, CDotFeatures* train_features, 
     CLabels* train_labels) :
	CLinearMachine(), m_z(1.0)
{
	set_z(z);
	set_features(train_features);
	set_labels(train_labels);
	set_termination(slep_options::get_default_termination());
	set_regularization(slep_options::get_default_regularization());
	set_tolerance(slep_options::get_default_tolerance());
	set_max_iter(slep_options::get_default_max_iter());
}

CSLEPMachine::~CSLEPMachine()
{
}

void CSLEPMachine::register_parameters()
{
	SG_ADD(&m_z, "z", "regularization coefficient", MS_AVAILABLE);
	SG_ADD(&m_termination, "termination", "termination", MS_NOT_AVAILABLE);
	SG_ADD(&m_regularization, "regularization", "regularization", MS_NOT_AVAILABLE);
	SG_ADD(&m_tolerance, "tolerance", "tolerance", MS_NOT_AVAILABLE);
	SG_ADD(&m_max_iter, "max_iter", "maximum number of iterations", MS_NOT_AVAILABLE);
}

int32_t CSLEPMachine::get_max_iter() const
{
	return m_max_iter;
}
int32_t CSLEPMachine::get_regularization() const
{
	return m_regularization;
}
int32_t CSLEPMachine::get_termination() const
{
	return m_termination;
}
float64_t CSLEPMachine::get_tolerance() const
{
	return m_tolerance;
}
float64_t CSLEPMachine::get_z() const
{
	return m_z;
}

void CSLEPMachine::set_max_iter(int32_t max_iter)
{
	ASSERT(max_iter>=0);
	m_max_iter = max_iter;
}
void CSLEPMachine::set_regularization(int32_t regularization)
{
	ASSERT(regularization==0 || regularization==1);
	m_regularization = regularization;
}
void CSLEPMachine::set_termination(int32_t termination)
{
	ASSERT(termination>=0 && termination<=4);
	m_termination = termination;
}
void CSLEPMachine::set_tolerance(float64_t tolerance)
{
	ASSERT(tolerance>0.0);
	m_tolerance = tolerance;
}
void CSLEPMachine::set_z(float64_t z)
{
	m_z = z;
}
}
