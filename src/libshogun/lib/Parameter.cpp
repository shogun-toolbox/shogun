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

TParameter::TParameter(const TSGDataType* datatype, void* parameter,
					   const char* name, const char* description)
	:m_datatype(*datatype)
{
	m_parameter = parameter;
	m_name = strdup(name);
	m_description = strdup(description);

	CSGObject** p = (CSGObject**) m_parameter;
	if(is_sgobject()) SG_REF(*p);
}

TParameter::~TParameter(void)
{
	CSGObject** p = (CSGObject**) m_parameter;
	if(is_sgobject()) SG_UNREF(*p);

	free(m_description); free(m_name);
}

bool
TParameter::is_sgobject(void)
{
	return m_datatype.m_ptype == PT_SGOBJECT_PTR
		|| m_datatype.m_ctype != CT_SCALAR;
}

CParameter::CParameter(void) :m_parameters()
{
}

CParameter::~CParameter(void)
{
	for (int32_t i=0; i<get_num_parameters(); i++)
		delete m_parameters.get_element(i);
}

void
CParameter::add_type(const TSGDataType* type, void* param,
					 const char* name, const char* description)
{
	m_parameters.append_element(
		new TParameter(type, param, name, description)
		);
}

void
CParameter::list_parameters()
{
	for (int32_t i=0; i<get_num_parameters(); i++)
		SG_PRINT("Parameter '%s'\n",
				 m_parameters.get_element(i)->m_name);
}
