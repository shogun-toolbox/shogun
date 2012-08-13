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
#include <shogun/structure/MulticlassSOLabels.h>

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
		CFeatures*      features)
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

SGVector< float64_t > CLinearStructuredOutputMachine::get_w() const
{
	return m_w;
}

CStructuredLabels* CLinearStructuredOutputMachine::apply_structured(CFeatures* data)
{
	if (data)
	{
		set_features(data);
	}

	CStructuredLabels* out;
	if ( !m_features )
	{
		out = new CStructuredLabels();
	}
	else
	{
		out = new CStructuredLabels( m_features->get_num_vectors() );
		for ( int32_t i = 0 ; i < m_features->get_num_vectors() ; ++i )
		{
			CResultSet* result = m_model->argmax(m_w, i, false);
			out->add_label(result->argmax);

			SG_UNREF(result);
		}
	}

	SG_REF(out);
	return out;
}

void CLinearStructuredOutputMachine::register_parameters()
{
	SG_ADD((CSGObject**)&m_features, "m_features", "Feature object", MS_NOT_AVAILABLE);
	SG_ADD(&m_w, "m_w", "Weight vector", MS_NOT_AVAILABLE);
}
