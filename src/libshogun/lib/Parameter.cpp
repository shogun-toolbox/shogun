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
}

TParameter::~TParameter(void)
{
	free(m_description); free(m_name);
}

bool
TParameter::is_sgserializable(void)
{
	return m_datatype.m_ptype == PT_SGSERIALIZABLE_PTR
		&& m_datatype.m_ctype == CT_SCALAR;
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
	char buf[50];
	m_datatype.to_string(buf);

	SG_PRINT("\n%s\n%35s %24s :%s\n", prefix, m_description == NULL
			 || *m_description == '\0' ? "(Parameter)": m_description,
			 m_name, buf);

	if (is_sgserializable()
		&& *(CSGSerializable**) m_parameter != NULL) {
		char* p = new_prefix(prefix, m_name);
		(*(CSGSerializable**) m_parameter)->print_serializable(p);
		free(p);
	}
}

bool
TParameter::save(CSerializableFile* file, const char* prefix)
{
	bool result;

	if (is_sgserializable()
		&& *(CSGSerializable**) m_parameter != NULL) {
		char* p = new_prefix(prefix, m_name);
		result
			= (*(CSGSerializable**) m_parameter)
			->save_serializable(file, p);
		free(p);
	} else
		result = file->write_type(&m_datatype, m_parameter, m_name,
								  prefix);

	return result;
}

bool
TParameter::load(CSerializableFile* file, const char* prefix)
{
	bool result;

	if (is_sgserializable()
		&& *(CSGSerializable**) m_parameter != NULL) {
		char* p = new_prefix(prefix, m_name);
		result
			= (*(CSGSerializable**) m_parameter)
			->load_serializable(file, p);
		free(p);
	} else
		result = file->read_type(&m_datatype, m_parameter, m_name,
								 prefix);

	return result;
}

CParameter::CParameter(CIO* io_) :m_params(io_)
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
	for (int32_t i=0; i<get_num_parameters(); i++)
		if (strcmp(m_params.get_element(i)->m_name, name) == 0) {
			SG_ERROR("FATAL: CParameter::add_type(): "
					 "Double parameter `%s'!", name);
			exit(1);
		}

	m_params.append_element(
		new TParameter(type, param, name, description)
		);
}

void
CParameter::print(const char* prefix)
{
	for (int32_t i=0; i<get_num_parameters(); i++)
		m_params.get_element(i)->print(io, prefix);
}

bool
CParameter::save(CSerializableFile* file, const char* prefix)
{
	for (int32_t i=0; i<get_num_parameters(); i++)
		if (!m_params.get_element(i)->save(file, prefix))
			return false;

	return true;
}

bool
CParameter::load(CSerializableFile* file, const char* prefix)
{
	for (int32_t i=0; i<get_num_parameters(); i++)
		if (!m_params.get_element(i)->load(file, prefix))
			return false;

	return true;
}
