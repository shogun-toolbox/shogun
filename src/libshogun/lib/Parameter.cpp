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
#include "base/class_list.h"

using namespace shogun;

/* **************************************************************** */
/* Scalar wrappers  */

void
CParameter::add(bool* param, const char* name,
				const char* description) {
	TSGDataType type(CT_SCALAR, PT_BOOL);
	add_type(&type, param, name, description);
}

void
CParameter::add(char* param, const char* name,
				const char* description) {
	TSGDataType type(CT_SCALAR, PT_CHAR);
	add_type(&type, param, name, description);
}

void
CParameter::add(int8_t* param, const char* name,
				const char* description) {
	TSGDataType type(CT_SCALAR, PT_INT8);
	add_type(&type, param, name, description);
}

void
CParameter::add(uint8_t* param, const char* name,
				const char* description) {
	TSGDataType type(CT_SCALAR, PT_UINT8);
	add_type(&type, param, name, description);
}

void
CParameter::add(int16_t* param, const char* name,
				const char* description) {
	TSGDataType type(CT_SCALAR, PT_INT16);
	add_type(&type, param, name, description);
}

void
CParameter::add(uint16_t* param, const char* name,
				const char* description) {
	TSGDataType type(CT_SCALAR, PT_UINT16);
	add_type(&type, param, name, description);
}

void
CParameter::add(int32_t* param, const char* name,
				const char* description) {
	TSGDataType type(CT_SCALAR, PT_INT32);
	add_type(&type, param, name, description);
}

void
CParameter::add(uint32_t* param, const char* name,
				const char* description) {
	TSGDataType type(CT_SCALAR, PT_UINT32);
	add_type(&type, param, name, description);
}

void
CParameter::add(int64_t* param, const char* name,
				const char* description) {
	TSGDataType type(CT_SCALAR, PT_INT64);
	add_type(&type, param, name, description);
}

void
CParameter::add(uint64_t* param, const char* name,
				const char* description) {
	TSGDataType type(CT_SCALAR, PT_UINT64);
	add_type(&type, param, name, description);
}

void
CParameter::add(float32_t* param, const char* name,
				const char* description) {
	TSGDataType type(CT_SCALAR, PT_FLOAT32);
	add_type(&type, param, name, description);
}

void
CParameter::add(float64_t* param, const char* name,
				const char* description) {
	TSGDataType type(CT_SCALAR, PT_FLOAT64);
	add_type(&type, param, name, description);
}

void
CParameter::add(floatmax_t* param, const char* name,
				const char* description) {
	TSGDataType type(CT_SCALAR, PT_FLOATMAX);
	add_type(&type, param, name, description);
}

void
CParameter::add(CSGSerializable** param,
				const char* name, const char* description) {
	TSGDataType type(CT_SCALAR, PT_SGSERIALIZABLE_PTR);
	add_type(&type, param, name, description);
}

/* **************************************************************** */
/* Vector wrappers  */

void
CParameter::add_vector(
	bool** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_BOOL, length);
	add_type(&type, param, name, description);
}

void
CParameter::add_vector(
	char** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_CHAR, length);
	add_type(&type, param, name, description);
}

void
CParameter::add_vector(
	int8_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_INT8, length);
	add_type(&type, param, name, description);
}

void
CParameter::add_vector(
	uint8_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_UINT8, length);
	add_type(&type, param, name, description);
}

void
CParameter::add_vector(
	int16_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_INT16, length);
	add_type(&type, param, name, description);
}

void
CParameter::add_vector(
	uint16_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_UINT16, length);
	add_type(&type, param, name, description);
}

void
CParameter::add_vector(
	int32_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_INT32, length);
	add_type(&type, param, name, description);
}

void
CParameter::add_vector(
	uint32_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_UINT32, length);
	add_type(&type, param, name, description);
}

void
CParameter::add_vector(
	int64_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_INT64, length);
	add_type(&type, param, name, description);
}

void
CParameter::add_vector(
	uint64_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_UINT64, length);
	add_type(&type, param, name, description);
}

void
CParameter::add_vector(
	float32_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_FLOAT32, length);
	add_type(&type, param, name, description);
}

void
CParameter::add_vector(
	float64_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_FLOAT64, length);
	add_type(&type, param, name, description);
}

void
CParameter::add_vector(
	floatmax_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_FLOATMAX, length);
	add_type(&type, param, name, description);
}

void
CParameter::add_vector(CSGSerializable*** param, index_t* length,
					   const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, PT_SGSERIALIZABLE_PTR, length);
	add_type(&type, param, name, description);
}

/* **************************************************************** */
/* Matrix wrappers  */

void
CParameter::add_matrix(
	bool** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_BOOL, length_y, length_x);
	add_type(&type, param, name, description);
}

void
CParameter::add_matrix(
	char** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_CHAR, length_y, length_x);
	add_type(&type, param, name, description);
}

void
CParameter::add_matrix(
	int8_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_INT8, length_y, length_x);
	add_type(&type, param, name, description);
}

void
CParameter::add_matrix(
	uint8_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_UINT8, length_y, length_x);
	add_type(&type, param, name, description);
}

void
CParameter::add_matrix(
	int16_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_INT16, length_y, length_x);
	add_type(&type, param, name, description);
}

void
CParameter::add_matrix(
	uint16_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_UINT16, length_y, length_x);
	add_type(&type, param, name, description);
}

void
CParameter::add_matrix(
	int32_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_INT32, length_y, length_x);
	add_type(&type, param, name, description);
}

void
CParameter::add_matrix(
	uint32_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_UINT32, length_y, length_x);
	add_type(&type, param, name, description);
}

void
CParameter::add_matrix(
	int64_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_INT64, length_y, length_x);
	add_type(&type, param, name, description);
}

void
CParameter::add_matrix(
	uint64_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_UINT64, length_y, length_x);
	add_type(&type, param, name, description);
}

void
CParameter::add_matrix(
	float32_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_FLOAT32, length_y, length_x);
	add_type(&type, param, name, description);
}

void
CParameter::add_matrix(
	float64_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_FLOAT64, length_y, length_x);
	add_type(&type, param, name, description);
}

void
CParameter::add_matrix(
	floatmax_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_FLOATMAX, length_y, length_x);
	add_type(&type, param, name, description);
}

void
CParameter::add_matrix(
	CSGSerializable*** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_SGSERIALIZABLE_PTR, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

/* **************************************************************** */
/* End of wrappers  */

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

	if (m_datatype.m_ptype == PT_SGSERIALIZABLE_PTR
		&& m_datatype.m_ctype == CT_SCALAR
		&& *(CSGSerializable**) m_parameter != NULL) {
		char* p = new_prefix(prefix, m_name);
		(*(CSGSerializable**) m_parameter)->print_serializable(p);
		free(p);
	}
}

void
TParameter::new_cont(index_t new_len_y, index_t new_len_x)
{
	if (*(void**) m_parameter != NULL) {
		index_t old_length = *m_datatype.m_length_y;
		switch (m_datatype.m_ctype) {
		case CT_MATRIX:
			old_length *= *m_datatype.m_length_x; break;
		case CT_SCALAR: case CT_VECTOR: break;
		}

		switch (m_datatype.m_ptype) {
		case PT_BOOL: delete[] *(bool**) m_parameter; break;
		case PT_CHAR: delete[] *(char**) m_parameter; break;
		case PT_INT8: delete[] *(int8_t**) m_parameter; break;
		case PT_UINT8: delete[] *(uint8_t**) m_parameter; break;
		case PT_INT16: delete[] *(int16_t**) m_parameter; break;
		case PT_UINT16: delete[] *(uint16_t**) m_parameter; break;
		case PT_INT32: delete[] *(int32_t**) m_parameter; break;
		case PT_UINT32: delete[] *(uint32_t**) m_parameter; break;
		case PT_INT64: delete[] *(int64_t**) m_parameter; break;
		case PT_UINT64: delete[] *(uint64_t**) m_parameter; break;
		case PT_FLOAT32: delete[] *(float32_t**) m_parameter; break;
		case PT_FLOAT64: delete[] *(float64_t**) m_parameter; break;
		case PT_FLOATMAX: delete[] *(floatmax_t**) m_parameter; break;
		case PT_SGSERIALIZABLE_PTR:
			CSGSerializable** buf = *(CSGSerializable***) m_parameter;
			for (index_t i=0; i<old_length; i++)
				if (buf[i] != NULL) delete buf[i];

			delete buf;
			break;
		}
	}

	*(void**) m_parameter = NULL;
	index_t new_length = new_len_y*new_len_x;
	if (new_length == 0) return;

	switch (m_datatype.m_ptype) {
	case PT_BOOL:
		*(bool**) m_parameter = new bool[new_length]; break;
	case PT_CHAR:
		*(char**) m_parameter = new char[new_length]; break;
	case PT_INT8:
		*(int8_t**) m_parameter = new int8_t[new_length]; break;
	case PT_UINT8:
		*(uint8_t**) m_parameter = new uint8_t[new_length]; break;
	case PT_INT16:
		*(int16_t**) m_parameter = new int16_t[new_length]; break;
	case PT_UINT16:
		*(uint16_t**) m_parameter = new uint16_t[new_length]; break;
	case PT_INT32:
		*(int32_t**) m_parameter = new int32_t[new_length]; break;
	case PT_UINT32:
		*(uint32_t**) m_parameter = new uint32_t[new_length]; break;
	case PT_INT64:
		*(int64_t**) m_parameter = new int64_t[new_length]; break;
	case PT_UINT64:
		*(uint64_t**) m_parameter = new uint64_t[new_length]; break;
	case PT_FLOAT32:
		*(float32_t**) m_parameter = new float32_t[new_length]; break;
	case PT_FLOAT64:
		*(float64_t**) m_parameter = new float64_t[new_length]; break;
	case PT_FLOATMAX:
		*(floatmax_t**) m_parameter = new floatmax_t[new_length]; break;
	case PT_SGSERIALIZABLE_PTR:
		*(CSGSerializable***) m_parameter
			= new CSGSerializable*[new_length]();
		break;
	}
}

bool
TParameter::new_sgserial(CIO* io, CSGSerializable** param,
						 const char* sgserializable_name,
						 const char* prefix)
{
	if (*param != NULL) delete *param;

	*param = new_sgserializable(sgserializable_name);

	if (*param == NULL) {
		SG_WARNING("FATAL: TParameter::new_sgserial(): "
				   "Class `C%s' was not listed during compiling Shogun"
				   " :( ...  Can not construct it for `%s%s'!",
				   sgserializable_name, prefix, m_name);

		return false;
	}

	return true;
}

bool
TParameter::save_scalar(CIO* io, CSerializableFile* file,
						const void* param, const char* prefix)
{
	bool result = true;

	if (m_datatype.m_ptype == PT_SGSERIALIZABLE_PTR) {
		const char* sgserial_name = *(CSGSerializable**) param == NULL
			?"" :(*(CSGSerializable**) param)->get_name();

		result &= file->write_sgserializable_begin(
			&m_datatype, m_name, prefix, sgserial_name);
		if (*sgserial_name != '\0') {
			char* p = new_prefix(prefix, m_name);
			result
				&= (*(CSGSerializable**) param)
				->save_serializable(file, p);
			free(p);
		}
		result &= file->write_sgserializable_end(
			&m_datatype, m_name, prefix, sgserial_name);
	} else
		result &= file->write_scalar(&m_datatype, m_name, prefix,
									 param);

	return result;
}

bool
TParameter::load_scalar(CIO* io, CSerializableFile* file,
						void* param, const char* prefix)
{
	bool result = true;

	if (m_datatype.m_ptype == PT_SGSERIALIZABLE_PTR) {
		char sgserial_name[256] = {'\0'};

		result &= file->read_sgserializable_begin(
			&m_datatype, m_name, prefix, sgserial_name);
		if (*sgserial_name != '\0') {
			if (!new_sgserial(io, (CSGSerializable**) param,
							  sgserial_name, prefix))
				return false;

			char* p = new_prefix(prefix, m_name);
			result
				&= (*(CSGSerializable**) param)
				->load_serializable(file, p);
			free(p);
		}
		result &= file->read_sgserializable_end(
			&m_datatype, m_name, prefix, sgserial_name);
	} else
		result &= file->read_scalar(&m_datatype, m_name, prefix,
									param);

	return result;
}

bool
TParameter::save(CIO* io, CSerializableFile* file, const char* prefix)
{
	bool result = true;

	result &= file->write_type_begin(&m_datatype, m_name, prefix);

	switch (m_datatype.m_ctype) {
	case CT_SCALAR:
		result &= save_scalar(io, file, m_parameter, prefix);
		break;
	case CT_VECTOR: case CT_MATRIX:
		index_t len_real_y = 0, len_real_x = 0;

		len_real_y = *m_datatype.m_length_y;
		if (*(void**) m_parameter == NULL && len_real_y != 0) {
			SG_WARNING("Inconsistence between data structure and "
					   "len_y during saving `%s%s'!  Continuing with "
					   "len_y=0.\n",
					   prefix, m_name);
			len_real_y = 0;
		}

		switch (m_datatype.m_ctype) {
		case CT_VECTOR:
			len_real_x = 1; break;
		case CT_MATRIX:
			len_real_x = *m_datatype.m_length_x;
			if (*(void**) m_parameter == NULL && len_real_x != 0) {
				SG_WARNING("Inconsistence between data structure and "
						   "len_x during saving `%s%s'.!  Continuing with "
						   "len_x=0.\n",
						   prefix, m_name);
				len_real_x = 0;
			}
			break;
		case CT_SCALAR: break;
		}

		result &= file->write_cont_begin(&m_datatype, m_name, prefix,
										 len_real_y, len_real_x);

		/* ******************************************************** */

		for (index_t x=0; x<len_real_x; x++)
			for (index_t y=0; y<len_real_y; y++) {
				result &= file->write_item_begin(
					&m_datatype, m_name, prefix, y, x);
				result &= save_scalar(
					io, file, (*(char**) m_parameter)
					+ (x*len_real_y + y)*m_datatype.sizeof_ptype(),
					prefix);
				result &= file->write_item_end(
					&m_datatype, m_name, prefix, y, x);
			}

		/* ******************************************************** */

		result &= file->write_cont_end(&m_datatype, m_name, prefix,
									   len_real_y, len_real_x);

		break;
	}

	result &= file->write_type_end(&m_datatype, m_name, prefix);

	return result;
}

bool
TParameter::load(CIO* io, CSerializableFile* file, const char* prefix)
{
	bool result = true;

	result &= file->read_type_begin(&m_datatype, m_name, prefix);

	switch (m_datatype.m_ctype) {
	case CT_SCALAR:
		result &= load_scalar(io, file, m_parameter, prefix);
		break;
	case CT_VECTOR: case CT_MATRIX:
		index_t len_read_y = 0, len_read_x = 0;

		result &= file->read_cont_begin(&m_datatype, m_name, prefix,
										&len_read_y, &len_read_x);

		/* ******************************************************** */

		/* Do not change the order!  NEW_CONT is accessing the members
		 * of M_DATATYPE.
		 */
		switch (m_datatype.m_ctype) {
		case CT_VECTOR:
			len_read_x = 1;
			new_cont(len_read_y, len_read_x);
			*m_datatype.m_length_y = len_read_y;
			break;
		case CT_MATRIX:
			new_cont(len_read_y, len_read_x);
			*m_datatype.m_length_y = len_read_y;
			*m_datatype.m_length_x = len_read_x;
			break;
		case CT_SCALAR: break;
		}

		for (index_t x=0; x<len_read_x; x++)
			for (index_t y=0; y<len_read_y; y++) {
				result &= file->read_item_begin(
					&m_datatype, m_name, prefix, y, x);
				result &= load_scalar(
					io, file, (*(char**) m_parameter)
					+ (x*len_read_y + y)*m_datatype.sizeof_ptype(),
					prefix);
				result &= file->read_item_end(
					&m_datatype, m_name, prefix, y, x);
			}

		/* ******************************************************** */

		result &= file->read_cont_end(&m_datatype, m_name, prefix,
									  len_read_y, len_read_x);

		break;
	}

	result &= file->read_type_end(&m_datatype, m_name, prefix);

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
	if (name == NULL || *name == '\0')
		SG_ERROR("FATAL: CParameter::add_type(): `name' is empty!");

	for (int32_t i=0; i<get_num_parameters(); i++)
		if (strcmp(m_params.get_element(i)->m_name, name) == 0)
			SG_ERROR("FATAL: CParameter::add_type(): "
					 "Double parameter `%s'!", name);

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
		if (!m_params.get_element(i)->save(io, file, prefix))
			return false;

	return true;
}

bool
CParameter::load(CSerializableFile* file, const char* prefix)
{
	for (int32_t i=0; i<get_num_parameters(); i++)
		if (!m_params.get_element(i)->load(io, file, prefix))
			return false;

	return true;
}
