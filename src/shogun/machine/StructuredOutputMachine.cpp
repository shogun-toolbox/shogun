/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/machine/StructuredOutputMachine.h>

using namespace shogun;

CStructuredOutputMachine::CStructuredOutputMachine()
: CMachine(), m_model(NULL)
{
	register_parameters();
}

CStructuredOutputMachine::CStructuredOutputMachine(
		CStructuredModel*  model,
		CStructuredLabels* labs)
: CMachine(), m_model(model)
{
	SG_REF(m_model);
	set_labels(labs);
	register_parameters();
}

CStructuredOutputMachine::~CStructuredOutputMachine()
{
	SG_UNREF(m_model);
}

void CStructuredOutputMachine::set_model(CStructuredModel* model)
{
	SG_UNREF(m_model);
	SG_REF(model);
	m_model = model;
}

CStructuredModel* CStructuredOutputMachine::get_model() const
{
	SG_REF(m_model);
	return m_model;
}

void CStructuredOutputMachine::register_parameters()
{
	SG_ADD((CSGObject**)&m_model, "m_model", "Structured model", MS_NOT_AVAILABLE);
}

void CStructuredOutputMachine::set_labels(CLabels* lab)
{
	CMachine::set_labels(lab);
	m_model->set_labels(CLabelsFactory::to_structured(lab));
}
