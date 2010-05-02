/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#include "lib/Parameter.h"

using namespace shogun;

void CParameter::add(float64_t* parameter, const char* name,
		CRange* range, const char* description)
{
	TParameter* par=new TParameter[1];
	par->parameter=parameter;
	par->datatype=DT_SCALAR_REAL;

	if (name)
		par->name=strdup(name);
	else
		par->name=NULL;

	if (description)
		par->description=strdup(description);
	else
		par->description=NULL;

	if (range)
		par->range=range;
	else
		par->range=NULL;

	m_parameters->append_element(par);
}

void CParameter::list_parameters()
{
	ASSERT(m_parameters);

	for (int32_t i=0; i<get_num_parameters(); i++)
	{
		TParameter* par=m_parameters->get_element(i);
		SG_PRINT("Parameter '%s'\n", par->name);
	}
}

void CParameter::free_parameters()
{
	ASSERT(m_parameters);

	for (int32_t i=0; i<get_num_parameters(); i++)
	{
		TParameter* par=m_parameters->get_element(i);
		free(par->name);
		free(par->description);
		delete[] par;
	}

	delete m_parameters;
	m_parameters=NULL;
}
