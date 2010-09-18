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

char*
TParameter::new_prefix(const char* s1, const char* s2)
{
	char tmp[256];

	snprintf(tmp, 256, "%s%s/", s1, s2);

	return strdup(tmp);
}

void
TParameter::print(CIO* io, const char* prefix)
{
	SG_PRINT("\n%s\n%35s %24s :", prefix, *m_description == '\0'
			 ? "(Parameter)": m_description, m_name);

	switch(m_datatype.m_ctype) {
	case CT_SCALAR:
		break;
	case CT_VECTOR:
		SG_PRINT("Vector<");
		break;
	case CT_STRING:
		SG_PRINT("String<");
		break;
	}

	switch(m_datatype.m_ptype) {
	case PT_BOOL:
		SG_PRINT("bool");
		break;
	case PT_CHAR:
		SG_PRINT("char");
		break;
	case PT_INT16:
		SG_PRINT("int16");
		break;
	case PT_INT32:
		SG_PRINT("int32");
		break;
	case PT_INT64:
		SG_PRINT("int64");
		break;
	case PT_FLOAT32:
		SG_PRINT("float32");
		break;
	case PT_FLOAT64:
		SG_PRINT("float64");
		break;
	case PT_FLOATMAX:
		SG_PRINT("floatmax");
		break;
	case PT_SGOBJECT_PTR:
		SG_PRINT("SGObject*");
		if (m_datatype.m_ctype == CT_SCALAR
			&& *(CSGObject**) m_parameter != NULL) {
			SG_PRINT("\n");

			char* p = new_prefix(prefix, m_name);
			(*(CSGObject**) m_parameter)->params_list(p);
			free(p);
		}
		break;
	}

	switch(m_datatype.m_ctype) {
	case CT_SCALAR:
		break;
	case CT_VECTOR:
	case CT_STRING:
		SG_PRINT(">*");
		break;
	}

	SG_PRINT("\n");
}

CParameter::CParameter(CIO* io_) :m_params(io)
{
	io = io_;

	SG_REF(io);
}

CParameter::~CParameter(void)
{
	for (int32_t i=0; i<get_num_parameters(); i++)
		delete m_params.get_element(i);

	SG_UNREF(io);
}

void
CParameter::add_type(const TSGDataType* type, void* param,
					 const char* name, const char* description)
{
	m_params.append_element(
		new TParameter(type, param, name, description)
		);
}

void
CParameter::list(const char* prefix)
{
	for (int32_t i=0; i<get_num_parameters(); i++)
		m_params.get_element(i)->print(io, prefix);
}
