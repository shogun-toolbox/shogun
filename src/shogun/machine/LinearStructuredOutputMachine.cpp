/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/machine/LinearStructuredOutputMachine.h>

using namespace shogun;

CLinearStructuredOutputMachine::CLinearStructuredOutputMachine()
: CStructuredOutputMachine(), m_features(NULL)
{
	register_parameters();
}

CLinearStructuredOutputMachine::CLinearStructuredOutputMachine(
		CStructuredModel*  model, 
		CLossFunction*     loss, 
		CStructuredLabels* labs, 
		CFeatures*         features)
: CStructuredOutputMachine(model, loss, labs), m_features(NULL)
{
	set_features(features);
	register_parameters();
}

CLinearStructuredOutputMachine::~CLinearStructuredOutputMachine()
{
	SG_UNREF(m_features)
}

void CLinearStructuredOutputMachine::set_features(CFeatures* f)
{
	SG_REF(f);
	SG_UNREF(m_features);
	m_features = f;
}

CFeatures* CLinearStructuredOutputMachine::get_features() const
{
	SG_REF(m_features);
	return m_features;
}

void CLinearStructuredOutputMachine::register_parameters()
{
	SG_ADD((CSGObject**)&m_features, "m_features", "Feature object", MS_NOT_AVAILABLE);
}
