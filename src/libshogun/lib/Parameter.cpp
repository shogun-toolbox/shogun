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

namespace shogun {

/* **************************************************************** */
/* Scalar wrappers  */

template <> void
CParameter::add<bool>(bool* param, const char* name,
					  const char* description) {
	TSGDataType type(CT_SCALAR, PT_BOOL);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add<char>(char* param, const char* name,
					  const char* description) {
	TSGDataType type(CT_SCALAR, PT_CHAR);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add<int8_t>(int8_t* param, const char* name,
						const char* description) {
	TSGDataType type(CT_SCALAR, PT_INT8);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add<uint8_t>(uint8_t* param, const char* name,
						 const char* description) {
	TSGDataType type(CT_SCALAR, PT_UINT8);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add<int16_t>(int16_t* param, const char* name,
						 const char* description) {
	TSGDataType type(CT_SCALAR, PT_INT16);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add<uint16_t>(uint16_t* param, const char* name,
						  const char* description) {
	TSGDataType type(CT_SCALAR, PT_UINT16);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add<int32_t>(int32_t* param, const char* name,
						 const char* description) {
	TSGDataType type(CT_SCALAR, PT_INT32);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add<uint32_t>(uint32_t* param, const char* name,
						  const char* description) {
	TSGDataType type(CT_SCALAR, PT_UINT32);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add<int64_t>(int64_t* param, const char* name,
						 const char* description) {
	TSGDataType type(CT_SCALAR, PT_INT64);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add<uint64_t>(uint64_t* param, const char* name,
						  const char* description) {
	TSGDataType type(CT_SCALAR, PT_UINT64);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add<float32_t>(float32_t* param, const char* name,
						   const char* description) {
	TSGDataType type(CT_SCALAR, PT_FLOAT32);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add<float64_t>(float64_t* param, const char* name,
						   const char* description) {
	TSGDataType type(CT_SCALAR, PT_FLOAT64);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add<floatmax_t>(floatmax_t* param, const char* name,
							const char* description) {
	TSGDataType type(CT_SCALAR, PT_FLOATMAX);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add<CSGSerializable*>(
	CSGSerializable** param, const char* name,
	const char* description) {
	TSGDataType type(CT_SCALAR, PT_SGSERIALIZABLE_PTR);
	add_type(&type, param, name, description);
}

/* **************************************************************** */
/* Vector wrappers  */

template <> void
CParameter::add_vector<bool>(
	bool** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_BOOL, length);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_vector<char>(
	char** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_CHAR, length);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_vector<int8_t>(
	int8_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_INT8, length);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_vector<uint8_t>(
	uint8_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_UINT8, length);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_vector<int16_t>(
	int16_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_INT16, length);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_vector<uint16_t>(
	uint16_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_UINT16, length);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_vector<int32_t>(
	int32_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_INT32, length);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_vector<uint32_t>(
	uint32_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_UINT32, length);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_vector<int64_t>(
	int64_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_INT64, length);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_vector<uint64_t>(
	uint64_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_UINT64, length);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_vector<float32_t>(
	float32_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_FLOAT32, length);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_vector<float64_t>(
	float64_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_FLOAT64, length);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_vector<floatmax_t>(
	floatmax_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_FLOATMAX, length);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_vector<CSGSerializable>(
	CSGSerializable** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, PT_SGSERIALIZABLE_PTR, length);
	add_type(&type, param, name, description);
}

/* **************************************************************** */
/* Matrix wrappers  */

template <> void
CParameter::add_matrix<bool>(
	bool** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_BOOL, length_y, length_x);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_matrix<char>(
	char** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_CHAR, length_y, length_x);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_matrix<int8_t>(
	int8_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_INT8, length_y, length_x);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_matrix<uint8_t>(
	uint8_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_UINT8, length_y, length_x);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_matrix<int16_t>(
	int16_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_INT16, length_y, length_x);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_matrix<uint16_t>(
	uint16_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_UINT16, length_y, length_x);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_matrix<int32_t>(
	int32_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_INT32, length_y, length_x);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_matrix<uint32_t>(
	uint32_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_UINT32, length_y, length_x);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_matrix<int64_t>(
	int64_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_INT64, length_y, length_x);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_matrix<uint64_t>(
	uint64_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_UINT64, length_y, length_x);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_matrix<float32_t>(
	float32_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_FLOAT32, length_y, length_x);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_matrix<float64_t>(
	float64_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_FLOAT64, length_y, length_x);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_matrix<floatmax_t>(
	floatmax_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_FLOATMAX, length_y, length_x);
	add_type(&type, param, name, description);
}

template <> void
CParameter::add_matrix<CSGSerializable>(
	CSGSerializable** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, PT_SGSERIALIZABLE_PTR, length_y, length_x);
	add_type(&type, param, name, description);
}

} /* namespace shogun  */


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
	if (name == NULL || *name == '\0')
		SG_ERROR("FATAL: CParameter::add_type(): `name' is empty!");

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
