/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include "base/Parameter.h"
#include "base/class_list.h"

using namespace shogun;

extern IO* sg_io;

/* **************************************************************** */
/* Scalar wrappers  */

void
Parameter::add(bool* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_BOOL);
	add_type(&type, param, name, description);
}

void
Parameter::add(char* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_CHAR);
	add_type(&type, param, name, description);
}

void
Parameter::add(int8_t* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT8);
	add_type(&type, param, name, description);
}

void
Parameter::add(uint8_t* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT8);
	add_type(&type, param, name, description);
}

void
Parameter::add(int16_t* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT16);
	add_type(&type, param, name, description);
}

void
Parameter::add(uint16_t* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT16);
	add_type(&type, param, name, description);
}

void
Parameter::add(int32_t* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(uint32_t* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(int64_t* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_INT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(uint64_t* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_UINT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(float32_t* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOAT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(float64_t* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOAT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(floatmax_t* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_FLOATMAX);
	add_type(&type, param, name, description);
}

void
Parameter::add(CSGObject** param,
			   const char* name, const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_SGOBJECT);
	add_type(&type, param, name, description);
}

void
Parameter::add(TString<bool>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_BOOL);
	add_type(&type, param, name, description);
}

void
Parameter::add(TString<char>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_CHAR);
	add_type(&type, param, name, description);
}

void
Parameter::add(TString<int8_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_INT8);
	add_type(&type, param, name, description);
}

void
Parameter::add(TString<uint8_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_UINT8);
	add_type(&type, param, name, description);
}

void
Parameter::add(TString<int16_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_INT16);
	add_type(&type, param, name, description);
}

void
Parameter::add(TString<uint16_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_UINT16);
	add_type(&type, param, name, description);
}

void
Parameter::add(TString<int32_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_INT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(TString<uint32_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_UINT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(TString<int64_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_INT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(TString<uint64_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_UINT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(TString<float32_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_FLOAT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(TString<float64_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_FLOAT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(TString<floatmax_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_FLOATMAX);
	add_type(&type, param, name, description);
}

void
Parameter::add(TSparse<bool>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_BOOL);
	add_type(&type, param, name, description);
}

void
Parameter::add(TSparse<char>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_CHAR);
	add_type(&type, param, name, description);
}

void
Parameter::add(TSparse<int8_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_INT8);
	add_type(&type, param, name, description);
}

void
Parameter::add(TSparse<uint8_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_UINT8);
	add_type(&type, param, name, description);
}

void
Parameter::add(TSparse<int16_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_INT16);
	add_type(&type, param, name, description);
}

void
Parameter::add(TSparse<uint16_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_UINT16);
	add_type(&type, param, name, description);
}

void
Parameter::add(TSparse<int32_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_INT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(TSparse<uint32_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_UINT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(TSparse<int64_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_INT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(TSparse<uint64_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_UINT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(TSparse<float32_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_FLOAT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(TSparse<float64_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_FLOAT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(TSparse<floatmax_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_FLOATMAX);
	add_type(&type, param, name, description);
}

/* **************************************************************** */
/* Vector wrappers  */

void
Parameter::add_vector(
	bool** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_BOOL, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(
	char** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_CHAR, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(
	int8_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_INT8, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(
	uint8_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_UINT8, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(
	int16_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_INT16, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(
	uint16_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_UINT16, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(
	int32_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_INT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(
	uint32_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_UINT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(
	int64_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_INT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(
	uint64_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_UINT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(
	float32_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_FLOAT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(
	float64_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_FLOAT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(
	floatmax_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_FLOATMAX, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(CSGObject*** param, index_t* length,
					   const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_SGOBJECT,
					 length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TString<bool>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_BOOL, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TString<char>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_CHAR, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TString<int8_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_INT8, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TString<uint8_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_UINT8, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TString<int16_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_INT16, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TString<uint16_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_UINT16, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TString<int32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_INT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TString<uint32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_UINT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TString<int64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_INT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TString<uint64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_UINT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TString<float32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_FLOAT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TString<float64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_FLOAT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TString<floatmax_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_FLOATMAX, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TSparse<bool>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_BOOL, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TSparse<char>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_CHAR, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TSparse<int8_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_INT8, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TSparse<uint8_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_UINT8, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TSparse<int16_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_INT16, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TSparse<uint16_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_UINT16, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TSparse<int32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_INT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TSparse<uint32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_UINT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TSparse<int64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_INT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TSparse<uint64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_UINT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TSparse<float32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_FLOAT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TSparse<float64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_FLOAT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(TSparse<floatmax_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_FLOATMAX, length);
	add_type(&type, param, name, description);
}

/* **************************************************************** */
/* Matrix wrappers  */

void
Parameter::add_matrix(
	bool** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_BOOL, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(
	char** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_CHAR, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(
	int8_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_INT8, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(
	uint8_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_UINT8, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(
	int16_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_INT16, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(
	uint16_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_UINT16, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(
	int32_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_INT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(
	uint32_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_UINT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(
	int64_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_INT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(
	uint64_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_UINT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(
	float32_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_FLOAT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(
	float64_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_FLOAT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(
	floatmax_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_FLOATMAX, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(
	CSGObject*** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_SGOBJECT,
					 length_y, length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TString<bool>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_BOOL, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TString<char>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_CHAR, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TString<int8_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_INT8, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TString<uint8_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_UINT8, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TString<int16_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_INT16, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TString<uint16_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_UINT16, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TString<int32_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_INT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TString<uint32_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_UINT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TString<int64_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_INT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TString<uint64_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_UINT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TString<float32_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_FLOAT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TString<float64_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_FLOAT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TString<floatmax_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_FLOATMAX, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TSparse<bool>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_BOOL, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TSparse<char>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_CHAR, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TSparse<int8_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_INT8, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TSparse<uint8_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_UINT8, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TSparse<int16_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_INT16, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TSparse<uint16_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_UINT16, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TSparse<int32_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_INT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TSparse<uint32_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_UINT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TSparse<int64_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_INT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TSparse<uint64_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_UINT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TSparse<float32_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_FLOAT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TSparse<float64_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_FLOAT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(TSparse<floatmax_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_FLOATMAX, length_y,
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
	char* tmp = new char[strlen(s1)+strlen(s2)+2];

	sprintf(tmp, "%s%s/", s1, s2);

	return tmp;
}

void
TParameter::print(const char* prefix)
{
	string_t buf;
	m_datatype.to_string(buf, STRING_LEN);

	SG_SPRINT("\n%s\n%35s %24s :%s\n", prefix, m_description == NULL
			 || *m_description == '\0' ? "(Parameter)": m_description,
			 m_name, buf);

	if (m_datatype.m_ptype == PT_SGOBJECT
		&& m_datatype.m_stype == ST_NONE
		&& m_datatype.m_ctype == CT_SCALAR
		&& *(CSGObject**) m_parameter != NULL) {
		char* p = new_prefix(prefix, m_name);
		(*(CSGObject**) m_parameter)->print_serializable(p);
		delete p;
	}
}

void
TParameter::delete_cont(void)
{
	if (*(void**) m_parameter != NULL) {
		index_t old_length = *m_datatype.m_length_y;
		switch (m_datatype.m_ctype) {
		case CT_MATRIX:
			old_length *= *m_datatype.m_length_x; break;
		case CT_SCALAR: case CT_VECTOR: break;
		}

		switch (m_datatype.m_stype) {
		case ST_NONE:
			switch (m_datatype.m_ptype) {
			case PT_BOOL:
				delete[] *(bool**) m_parameter; break;
			case PT_CHAR:
				delete[] *(char**) m_parameter; break;
			case PT_INT8:
				delete[] *(int8_t**) m_parameter; break;
			case PT_UINT8:
				delete[] *(uint8_t**) m_parameter; break;
			case PT_INT16:
				delete[] *(int16_t**) m_parameter; break;
			case PT_UINT16:
				delete[] *(uint16_t**) m_parameter; break;
			case PT_INT32:
				delete[] *(int32_t**) m_parameter; break;
			case PT_UINT32:
				delete[] *(uint32_t**) m_parameter; break;
			case PT_INT64:
				delete[] *(int64_t**) m_parameter; break;
			case PT_UINT64:
				delete[] *(uint64_t**) m_parameter; break;
			case PT_FLOAT32:
				delete[] *(float32_t**) m_parameter; break;
			case PT_FLOAT64:
				delete[] *(float64_t**) m_parameter; break;
			case PT_FLOATMAX:
				delete[] *(floatmax_t**) m_parameter; break;
			case PT_SGOBJECT:
				CSGObject** buf =
					*(CSGObject***) m_parameter;
				for (index_t i=0; i<old_length; i++)
					if (buf[i] != NULL) SG_UNREF(buf[i]);
				delete buf;
				break;
			}
			break;
		case ST_STRING:
			for (index_t i=0; i<old_length; i++) {
				TString<char>* buf = (TString<char>*) (*(char**)
						m_parameter + i *m_datatype.sizeof_stype());
				if (buf->length > 0) delete[] buf->string;
			}

			switch (m_datatype.m_ptype) {
			case PT_BOOL:
				delete[] *(TString<bool>**) m_parameter; break;
			case PT_CHAR:
				delete[] *(TString<char>**) m_parameter; break;
			case PT_INT8:
				delete[] *(TString<int8_t>**) m_parameter; break;
			case PT_UINT8:
				delete[] *(TString<uint8_t>**) m_parameter; break;
			case PT_INT16:
				delete[] *(TString<int16_t>**) m_parameter; break;
			case PT_UINT16:
				delete[] *(TString<uint16_t>**) m_parameter; break;
			case PT_INT32:
				delete[] *(TString<int32_t>**) m_parameter; break;
			case PT_UINT32:
				delete[] *(TString<uint32_t>**) m_parameter; break;
			case PT_INT64:
				delete[] *(TString<int64_t>**) m_parameter; break;
			case PT_UINT64:
				delete[] *(TString<uint64_t>**) m_parameter; break;
			case PT_FLOAT32:
				delete[] *(TString<float32_t>**) m_parameter; break;
			case PT_FLOAT64:
				delete[] *(TString<float64_t>**) m_parameter; break;
			case PT_FLOATMAX:
				delete[] *(TString<floatmax_t>**) m_parameter; break;
			case PT_SGOBJECT:
				SG_SERROR("TParameter::delete_cont(): Implementation "
						 "error: Could not delete "
						 "String<SGSerializable*>");
				break;
			}
			break;
		case ST_SPARSE:
			for (index_t i=0; i<old_length; i++) {
				TSparse<char>* buf = (TSparse<char>*) (*(char**)
						m_parameter + i *m_datatype.sizeof_stype());
				if (buf->num_feat_entries > 0) delete[] buf->features;
			}

			switch (m_datatype.m_ptype) {
			case PT_BOOL:
				delete[] *(TSparse<bool>**) m_parameter; break;
			case PT_CHAR:
				delete[] *(TSparse<char>**) m_parameter; break;
			case PT_INT8:
				delete[] *(TSparse<int8_t>**) m_parameter; break;
			case PT_UINT8:
				delete[] *(TSparse<uint8_t>**) m_parameter; break;
			case PT_INT16:
				delete[] *(TSparse<int16_t>**) m_parameter; break;
			case PT_UINT16:
				delete[] *(TSparse<uint16_t>**) m_parameter; break;
			case PT_INT32:
				delete[] *(TSparse<int32_t>**) m_parameter; break;
			case PT_UINT32:
				delete[] *(TSparse<uint32_t>**) m_parameter; break;
			case PT_INT64:
				delete[] *(TSparse<int64_t>**) m_parameter; break;
			case PT_UINT64:
				delete[] *(TSparse<uint64_t>**) m_parameter; break;
			case PT_FLOAT32:
				delete[] *(TSparse<float32_t>**) m_parameter; break;
			case PT_FLOAT64:
				delete[] *(TSparse<float64_t>**) m_parameter; break;
			case PT_FLOATMAX:
				delete[] *(TSparse<floatmax_t>**) m_parameter; break;
			case PT_SGOBJECT:
				SG_SERROR("TParameter::delete_cont(): Implementation "
						 "error: Could not delete "
						 "Sparse<SGSerializable*>");
				break;
			}
			break;
		} /* switch (m_datatype.m_stype)  */
	} /* if (*(void**) m_parameter != NULL)  */

	*(void**) m_parameter = NULL;
}

void
TParameter::new_cont(index_t new_len_y, index_t new_len_x)
{
	delete_cont();

	index_t new_length = new_len_y*new_len_x;
	if (new_length == 0) return;

	switch (m_datatype.m_stype) {
	case ST_NONE:
		switch (m_datatype.m_ptype) {
		case PT_BOOL:
			*(bool**) m_parameter
				= new bool[new_length]; break;
		case PT_CHAR:
			*(char**) m_parameter
				= new char[new_length]; break;
		case PT_INT8:
			*(int8_t**) m_parameter
				= new int8_t[new_length]; break;
		case PT_UINT8:
			*(uint8_t**) m_parameter
				= new uint8_t[new_length]; break;
		case PT_INT16:
			*(int16_t**) m_parameter
				= new int16_t[new_length]; break;
		case PT_UINT16:
			*(uint16_t**) m_parameter
				= new uint16_t[new_length]; break;
		case PT_INT32:
			*(int32_t**) m_parameter
				= new int32_t[new_length]; break;
		case PT_UINT32:
			*(uint32_t**) m_parameter
				= new uint32_t[new_length]; break;
		case PT_INT64:
			*(int64_t**) m_parameter
				= new int64_t[new_length]; break;
		case PT_UINT64:
			*(uint64_t**) m_parameter
				= new uint64_t[new_length]; break;
		case PT_FLOAT32:
			*(float32_t**) m_parameter
				= new float32_t[new_length]; break;
		case PT_FLOAT64:
			*(float64_t**) m_parameter
				= new float64_t[new_length]; break;
		case PT_FLOATMAX:
			*(floatmax_t**) m_parameter
				= new floatmax_t[new_length]; break;
		case PT_SGOBJECT:
			*(CSGObject***) m_parameter
				= new CSGObject*[new_length]();
			break;
		}
		break;
	case ST_STRING:
		switch (m_datatype.m_ptype) {
		case PT_BOOL:
			*(TString<bool>**) m_parameter
				= new TString<bool>[new_length]; break;
		case PT_CHAR:
			*(TString<char>**) m_parameter
				= new TString<char>[new_length]; break;
		case PT_INT8:
			*(TString<int8_t>**) m_parameter
				= new TString<int8_t>[new_length]; break;
		case PT_UINT8:
			*(TString<uint8_t>**) m_parameter
				= new TString<uint8_t>[new_length]; break;
		case PT_INT16:
			*(TString<int16_t>**) m_parameter
				= new TString<int16_t>[new_length]; break;
		case PT_UINT16:
			*(TString<uint16_t>**) m_parameter
				= new TString<uint16_t>[new_length]; break;
		case PT_INT32:
			*(TString<int32_t>**) m_parameter
				= new TString<int32_t>[new_length]; break;
		case PT_UINT32:
			*(TString<uint32_t>**) m_parameter
				= new TString<uint32_t>[new_length]; break;
		case PT_INT64:
			*(TString<int64_t>**) m_parameter
				= new TString<int64_t>[new_length]; break;
		case PT_UINT64:
			*(TString<uint64_t>**) m_parameter
				= new TString<uint64_t>[new_length]; break;
		case PT_FLOAT32:
			*(TString<float32_t>**) m_parameter
				= new TString<float32_t>[new_length]; break;
		case PT_FLOAT64:
			*(TString<float64_t>**) m_parameter
				= new TString<float64_t>[new_length]; break;
		case PT_FLOATMAX:
			*(TString<floatmax_t>**) m_parameter
				= new TString<floatmax_t>[new_length]; break;
		case PT_SGOBJECT:
			SG_SERROR("TParameter::new_cont(): Implementation "
					 "error: Could not allocate "
					 "String<SGSerializable*>");
			break;
		}
		memset(*(void**) m_parameter, 0, new_length
			   *m_datatype.sizeof_stype());
		break;
	case ST_SPARSE:
		switch (m_datatype.m_ptype) {
		case PT_BOOL:
			*(TSparse<bool>**) m_parameter
				= new TSparse<bool>[new_length]; break;
		case PT_CHAR:
			*(TSparse<char>**) m_parameter
				= new TSparse<char>[new_length]; break;
		case PT_INT8:
			*(TSparse<int8_t>**) m_parameter
				= new TSparse<int8_t>[new_length]; break;
		case PT_UINT8:
			*(TSparse<uint8_t>**) m_parameter
				= new TSparse<uint8_t>[new_length]; break;
		case PT_INT16:
			*(TSparse<int16_t>**) m_parameter
				= new TSparse<int16_t>[new_length]; break;
		case PT_UINT16:
			*(TSparse<uint16_t>**) m_parameter
				= new TSparse<uint16_t>[new_length]; break;
		case PT_INT32:
			*(TSparse<int32_t>**) m_parameter
				= new TSparse<int32_t>[new_length]; break;
		case PT_UINT32:
			*(TSparse<uint32_t>**) m_parameter
				= new TSparse<uint32_t>[new_length]; break;
		case PT_INT64:
			*(TSparse<int64_t>**) m_parameter
				= new TSparse<int64_t>[new_length]; break;
		case PT_UINT64:
			*(TSparse<uint64_t>**) m_parameter
				= new TSparse<uint64_t>[new_length]; break;
		case PT_FLOAT32:
			*(TSparse<float32_t>**) m_parameter
				= new TSparse<float32_t>[new_length]; break;
		case PT_FLOAT64:
			*(TSparse<float64_t>**) m_parameter
				= new TSparse<float64_t>[new_length]; break;
		case PT_FLOATMAX:
			*(TSparse<floatmax_t>**) m_parameter
				= new TSparse<floatmax_t>[new_length]; break;
		case PT_SGOBJECT:
			SG_SERROR("TParameter::new_cont(): Implementation "
					 "error: Could not allocate "
					 "Sparse<SGSerializable*>");
			break;
		}
		memset(*(void**) m_parameter, 0, new_length
			   *m_datatype.sizeof_stype());
		break;
	} /* switch (m_datatype.m_stype)  */
}

bool
TParameter::new_sgserial(CSGObject** param,
						 EPrimitiveType generic,
						 const char* sgserializable_name,
						 const char* prefix)
{
	if (*param != NULL)
		SG_UNREF(*param);

	*param = new_sgserializable(sgserializable_name, generic);

	if (*param == NULL) {
		string_t buf = {'\0'};

		if (generic != PT_NOT_GENERIC) {
			buf[0] = '<';
			TSGDataType::ptype_to_string(buf+1, generic,
										 STRING_LEN - 3);
			strcat(buf, ">");
		}

		SG_SWARNING("TParameter::new_sgserial(): "
				   "Class `C%s%s' was not listed during compiling Shogun"
				   " :( ...  Can not construct it for `%s%s'!",
				   sgserializable_name, buf, prefix, m_name);

		return false;
	}

	SG_REF(*param);
	return true;
}

bool
TParameter::save_ptype(CSerializableFile* file, const void* param,
					   const char* prefix)
{
	if (m_datatype.m_ptype == PT_SGOBJECT) {
		const char* sgserial_name = "";
		EPrimitiveType generic = PT_NOT_GENERIC;

		if (*(CSGObject**) param != NULL) {
			sgserial_name = (*(CSGObject**) param)->get_name();
			(*(CSGObject**) param)->is_generic(&generic);
		}

		if (!file->write_sgserializable_begin(
				&m_datatype, m_name, prefix, sgserial_name, generic))
			return false;
		if (*sgserial_name != '\0') {
			char* p = new_prefix(prefix, m_name);
			bool result = (*(CSGObject**) param)
				->save_serializable(file, p);
			delete p;
			if (!result) return false;
		}
		if (!file->write_sgserializable_end(
				&m_datatype, m_name, prefix, sgserial_name, generic))
			return false;
	} else
		if (!file->write_scalar(&m_datatype, m_name, prefix,
								param)) return false;

	return true;
}

bool
TParameter::load_ptype(CSerializableFile* file, void* param,
					   const char* prefix)
{
	if (m_datatype.m_ptype == PT_SGOBJECT) {
		string_t sgserial_name = {'\0'};
		EPrimitiveType generic = PT_NOT_GENERIC;

		if (!file->read_sgserializable_begin(
				&m_datatype, m_name, prefix, sgserial_name, &generic))
			return false;
		if (*sgserial_name != '\0') {
			if (!new_sgserial((CSGObject**) param, generic,
							  sgserial_name, prefix))
				return false;

			char* p = new_prefix(prefix, m_name);
			bool result = (*(CSGObject**) param)
				->load_serializable(file, p);
			delete p;
			if (!result) return false;
		}
		if (!file->read_sgserializable_end(
				&m_datatype, m_name, prefix, sgserial_name, generic))
			return false;
	} else
		if (!file->read_scalar(&m_datatype, m_name, prefix,
							   param)) return false;

	return true;
}

bool
TParameter::save_stype(CSerializableFile* file, const void* param,
					   const char* prefix)
{
	TString<char>* str_ptr = (TString<char>*) param;
	TSparse<char>* spr_ptr = (TSparse<char>*) param;
	index_t len_real;

	switch (m_datatype.m_stype) {
	case ST_NONE:
		if (!save_ptype(file, param, prefix)) return false;
		break;
	case ST_STRING:
		len_real = str_ptr->length;
		if (str_ptr->string == NULL && len_real != 0) {
			SG_SWARNING("Inconsistency between data structure and "
					   "len during saving string `%s%s'!  Continuing"
					   " with len=0.\n",
					   prefix, m_name);
			len_real = 0;
		}
		if (!file->write_string_begin(
				&m_datatype, m_name, prefix, len_real)) return false;
		for (index_t i=0; i<len_real; i++) {
			if (!file->write_stringentry_begin(
					&m_datatype, m_name, prefix, i)) return false;
			if (!save_ptype(file, (char*) str_ptr->string
							+ i *m_datatype.sizeof_ptype(), prefix))
				return false;
			if (!file->write_stringentry_end(
					&m_datatype, m_name, prefix, i)) return false;
		}
		if (!file->write_string_end(
				&m_datatype, m_name, prefix, len_real)) return false;
		break;
	case ST_SPARSE:
		len_real = spr_ptr->num_feat_entries;
		if (spr_ptr->features == NULL && len_real != 0) {
			SG_SWARNING("Inconsistency between data structure and "
					   "len during saving sparse `%s%s'!  Continuing"
					   " with len=0.\n",
					   prefix, m_name);
			len_real = 0;
		}
		if (!file->write_sparse_begin(
				&m_datatype, m_name, prefix, spr_ptr->vec_index,
				len_real)) return false;
		for (index_t i=0; i<len_real; i++) {
			TSparseEntry<char>* cur = (TSparseEntry<char>*)
				((char*) spr_ptr->features + i *TSGDataType
				 ::sizeof_sparseentry(m_datatype.m_ptype));
			if (!file->write_sparseentry_begin(
					&m_datatype, m_name, prefix, spr_ptr->features,
					cur->feat_index, i)) return false;
			if (!save_ptype(file, (char*) cur + TSGDataType
							::offset_sparseentry(m_datatype.m_ptype),
							prefix)) return false;
			if (!file->write_sparseentry_end(
					&m_datatype, m_name, prefix, spr_ptr->features,
					cur->feat_index, i)) return false;
		}
		if (!file->write_sparse_end(
				&m_datatype, m_name, prefix, spr_ptr->vec_index,
				len_real)) return false;
		break;
	}

	return true;
}

bool
TParameter::load_stype(CSerializableFile* file, void* param,
					   const char* prefix)
{
	TString<char>* str_ptr = (TString<char>*) param;
	TSparse<char>* spr_ptr = (TSparse<char>*) param;
	index_t len_real = 0;

	switch (m_datatype.m_stype) {
	case ST_NONE:
		if (!load_ptype(file, param, prefix)) return false;
		break;
	case ST_STRING:
		if (!file->read_string_begin(
				&m_datatype, m_name, prefix, &len_real))
			return false;
		str_ptr->string = len_real > 0
			? new char[len_real*m_datatype.sizeof_ptype()]: NULL;
		for (index_t i=0; i<len_real; i++) {
			if (!file->read_stringentry_begin(
					&m_datatype, m_name, prefix, i)) return false;
			if (!load_ptype(file, (char*) str_ptr->string
							+ i *m_datatype.sizeof_ptype(), prefix))
				return false;
			if (!file->read_stringentry_end(
					&m_datatype, m_name, prefix, i)) return false;
		}
		if (!file->read_string_end(
				&m_datatype, m_name, prefix, len_real))
			return false;
		str_ptr->length = len_real;
		break;
	case ST_SPARSE:
		if (!file->read_sparse_begin(
				&m_datatype, m_name, prefix, &spr_ptr->vec_index,
				&len_real)) return false;
		spr_ptr->features = len_real > 0? (TSparseEntry<char>*)
			new char[len_real *TSGDataType::sizeof_sparseentry(
				m_datatype.m_ptype)]: NULL;
		for (index_t i=0; i<len_real; i++) {
			TSparseEntry<char>* cur = (TSparseEntry<char>*)
				((char*) spr_ptr->features + i *TSGDataType
				 ::sizeof_sparseentry(m_datatype.m_ptype));
			if (!file->read_sparseentry_begin(
					&m_datatype, m_name, prefix, spr_ptr->features,
					&cur->feat_index, i)) return false;
			if (!load_ptype(file, (char*) cur + TSGDataType
							::offset_sparseentry(m_datatype.m_ptype),
							prefix)) return false;
			if (!file->read_sparseentry_end(
					&m_datatype, m_name, prefix, spr_ptr->features,
					&cur->feat_index, i)) return false;
		}
		if (!file->read_sparse_end(
				&m_datatype, m_name, prefix, &spr_ptr->vec_index,
				len_real)) return false;
		spr_ptr->num_feat_entries = len_real;
		break;
	}

	return true;
}

bool
TParameter::save(CSerializableFile* file, const char* prefix)
{
	const int32_t buflen=100;
	char* buf=new char[buflen];
	m_datatype.to_string(buf, buflen);
	SG_SDEBUG("Saving parameter '%s' of type '%s'\n", m_name, buf);
	delete[] buf;

	if (!file->write_type_begin(&m_datatype, m_name, prefix))
		return false;

	switch (m_datatype.m_ctype) {
	case CT_SCALAR:
		if (!save_stype(file, m_parameter, prefix)) return false;
		break;
	case CT_VECTOR: case CT_MATRIX:
		index_t len_real_y = 0, len_real_x = 0;

		len_real_y = *m_datatype.m_length_y;
		if (*(void**) m_parameter == NULL && len_real_y != 0) {
			SG_SWARNING("Inconsistency between data structure and "
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
				SG_SWARNING("Inconsistency between data structure and "
						   "len_x during saving `%s%s'!  Continuing "
						   "with len_x=0.\n",
						   prefix, m_name);
				len_real_x = 0;
			}

			if (len_real_x *len_real_y == 0)
				len_real_x = len_real_y = 0;

			break;
		case CT_SCALAR: break;
		}

		if (!file->write_cont_begin(&m_datatype, m_name, prefix,
									len_real_y, len_real_x))
			return false;

		/* ******************************************************** */

		for (index_t x=0; x<len_real_x; x++)
			for (index_t y=0; y<len_real_y; y++) {
				if (!file->write_item_begin(
						&m_datatype, m_name, prefix, y, x))
					return false;
				if (!save_stype(
						file, (*(char**) m_parameter)
						+ (x*len_real_y + y)*m_datatype.sizeof_stype(),
						prefix)) return false;
				if (!file->write_item_end(
						&m_datatype, m_name, prefix, y, x))
					return false;
			}

		/* ******************************************************** */

		if (!file->write_cont_end(&m_datatype, m_name, prefix,
								  len_real_y, len_real_x))
			return false;

		break;
	}

	if (!file->write_type_end(&m_datatype, m_name, prefix))
		return false;

	return true;
}

bool
TParameter::load(CSerializableFile* file, const char* prefix)
{
	const int32_t buflen=100;
	char* buf=new char[buflen];
	m_datatype.to_string(buf, buflen);
	SG_SDEBUG("Loading parameter '%s' of type '%s'\n", m_name, buf);
	delete[] buf;

	if (!file->read_type_begin(&m_datatype, m_name, prefix))
		return false;

	switch (m_datatype.m_ctype) {
	case CT_SCALAR:
		if (!load_stype(file, m_parameter, prefix)) return false;
		break;
	case CT_VECTOR: case CT_MATRIX:
		index_t len_read_y = 0, len_read_x = 0;

		if (!file->read_cont_begin(&m_datatype, m_name, prefix,
								   &len_read_y, &len_read_x))
			return false;

		/* ******************************************************** */

		switch (m_datatype.m_ctype) {
		case CT_VECTOR:
			len_read_x = 1;
			new_cont(len_read_y, len_read_x);
			break;
		case CT_MATRIX:
			new_cont(len_read_y, len_read_x);
			break;
		case CT_SCALAR: break;
		}

		for (index_t x=0; x<len_read_x; x++)
			for (index_t y=0; y<len_read_y; y++) {
				if (!file->read_item_begin(
						&m_datatype, m_name, prefix, y, x))
					return false;
				if (!load_stype(
						file, (*(char**) m_parameter)
						+ (x*len_read_y + y)*m_datatype.sizeof_stype(),
						prefix)) return false;
				if (!file->read_item_end(
						&m_datatype, m_name, prefix, y, x))
					return false;
			}

		switch (m_datatype.m_ctype) {
		case CT_VECTOR:
			*m_datatype.m_length_y = len_read_y;
			break;
		case CT_MATRIX:
			*m_datatype.m_length_y = len_read_y;
			*m_datatype.m_length_x = len_read_x;
			break;
		case CT_SCALAR: break;
		}

		/* ******************************************************** */

		if (!file->read_cont_end(&m_datatype, m_name, prefix,
								 len_read_y, len_read_x))
			return false;

		break;
	}

	if (!file->read_type_end(&m_datatype, m_name, prefix))
		return false;

	return true;
}

Parameter::Parameter(void)
{
	SG_REF(sg_io);
}

Parameter::~Parameter(void)
{
	for (int32_t i=0; i<get_num_parameters(); i++)
		delete m_params.get_element(i);

	SG_UNREF(sg_io);
}

void
Parameter::add_type(const TSGDataType* type, void* param,
					 const char* name, const char* description)
{
	if (name == NULL || *name == '\0')
		SG_SERROR("FATAL: Parameter::add_type(): `name' is empty!");

	for (int32_t i=0; i<get_num_parameters(); i++)
		if (strcmp(m_params.get_element(i)->m_name, name) == 0)
			SG_SERROR("FATAL: Parameter::add_type(): "
					 "Double parameter `%s'!", name);

	m_params.append_element(
		new TParameter(type, param, name, description)
		);
}

void
Parameter::print(const char* prefix)
{
	for (int32_t i=0; i<get_num_parameters(); i++)
		m_params.get_element(i)->print(prefix);
}

bool
Parameter::save(CSerializableFile* file, const char* prefix)
{
	for (int32_t i=0; i<get_num_parameters(); i++)
	{
		if (!m_params.get_element(i)->save(file, prefix))
			return false;
	}

	return true;
}

bool
Parameter::load(CSerializableFile* file, const char* prefix)
{
	for (int32_t i=0; i<get_num_parameters(); i++)
		if (!m_params.get_element(i)->load(file, prefix))
			return false;

	return true;
}
