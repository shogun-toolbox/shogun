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
: CMachine(), m_model(NULL), m_loss(NULL)
{
	register_parameters();
}

CStructuredOutputMachine::CStructuredOutputMachine(
		CStructuredModel*  model,
		CStructuredLoss*   loss,
		CStructuredLabels* labs)
: CMachine(), m_model(model), m_loss(loss)
{
	SG_REF(m_model);
	SG_REF(m_loss);
	set_labels(labs);
	register_parameters();
}

CStructuredOutputMachine::~CStructuredOutputMachine()
{
	SG_UNREF(m_model);
	SG_UNREF(m_loss);
}

// TODO
void CStructuredOutputMachine::set_labels(CStructuredLabels* labs)
{
	SG_ERROR("CStructuredOutputMachine::set_labels not implemented yet."
		 " Cause: CLabels and CStructuredLabels hierarchy.\n");
}

CLabels* CStructuredOutputMachine::apply()
{
	SG_NOTIMPLEMENTED;
	return NULL;
}

CLabels* CStructuredOutputMachine::apply(CFeatures* data)
{
	SG_NOTIMPLEMENTED;
	return NULL;
}

void CStructuredOutputMachine::register_parameters()
{
	SG_ADD((CSGObject**)&m_model, "m_model", "Structured model", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_loss, "m_loss", "Structured loss", MS_NOT_AVAILABLE);
}
