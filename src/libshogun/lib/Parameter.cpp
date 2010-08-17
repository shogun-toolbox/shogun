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

void CParameter::add_double(float64_t* parameter, const char* name,
			const char* description, float64_t min_value, float64_t max_value)
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

	par->min_value_float64=min_value;
	par->max_value_float64=max_value;

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
