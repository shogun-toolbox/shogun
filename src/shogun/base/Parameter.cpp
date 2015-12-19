/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Written (W) 2011-2013 Heiko Strathmann
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <string.h>

#include <shogun/base/Parameter.h>
#include <shogun/base/class_list.h>
#include <shogun/lib/Hash.h>
#include <shogun/lib/memory.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>

#include <shogun/lib/SGString.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/io/SerializableFile.h>

using namespace shogun;


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
Parameter::add(complex128_t* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_COMPLEX128);
	add_type(&type, param, name, description);
}

void
Parameter::add(CSGObject** param,
			   const char* name, const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_SGOBJECT);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGString<bool>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_BOOL);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGString<char>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_CHAR);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGString<int8_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_INT8);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGString<uint8_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_UINT8);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGString<int16_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_INT16);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGString<uint16_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_UINT16);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGString<int32_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_INT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGString<uint32_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_UINT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGString<int64_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_INT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGString<uint64_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_UINT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGString<float32_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_FLOAT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGString<float64_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_FLOAT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGString<floatmax_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_STRING, PT_FLOATMAX);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGSparseVector<bool>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_BOOL);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGSparseVector<char>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_CHAR);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGSparseVector<int8_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_INT8);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGSparseVector<uint8_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_UINT8);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGSparseVector<int16_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_INT16);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGSparseVector<uint16_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_UINT16);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGSparseVector<int32_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_INT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGSparseVector<uint32_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_UINT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGSparseVector<int64_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_INT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGSparseVector<uint64_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_UINT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGSparseVector<float32_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_FLOAT32);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGSparseVector<float64_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_FLOAT64);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGSparseVector<floatmax_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_FLOATMAX);
	add_type(&type, param, name, description);
}

void
Parameter::add(SGSparseVector<complex128_t>* param, const char* name,
			   const char* description) {
	TSGDataType type(CT_SCALAR, ST_SPARSE, PT_COMPLEX128);
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
Parameter::add_vector(
	complex128_t** param, index_t* length, const char* name,
	const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_COMPLEX128, length);
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
Parameter::add_vector(SGString<bool>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_BOOL, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGString<char>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_CHAR, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGString<int8_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_INT8, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGString<uint8_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_UINT8, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGString<int16_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_INT16, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGString<uint16_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_UINT16, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGString<int32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_INT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGString<uint32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_UINT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGString<int64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_INT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGString<uint64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_UINT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGString<float32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_FLOAT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGString<float64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_FLOAT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGString<floatmax_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_STRING, PT_FLOATMAX, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGSparseVector<bool>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_BOOL, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGSparseVector<char>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_CHAR, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGSparseVector<int8_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_INT8, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGSparseVector<uint8_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_UINT8, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGSparseVector<int16_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_INT16, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGSparseVector<uint16_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_UINT16, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGSparseVector<int32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_INT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGSparseVector<uint32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_UINT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGSparseVector<int64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_INT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGSparseVector<uint64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_UINT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGSparseVector<float32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_FLOAT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGSparseVector<float64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_FLOAT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGSparseVector<floatmax_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_FLOATMAX, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGSparseVector<complex128_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_SPARSE, PT_COMPLEX128, length);
	add_type(&type, param, name, description);
}




void Parameter::add(SGVector<bool>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_BOOL, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<char>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_CHAR, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<int8_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_INT8, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<uint8_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_UINT8, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<int16_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_INT16, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<uint16_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_UINT16, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<int32_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_INT32, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<uint32_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_UINT32, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<int64_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_INT64, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<uint64_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_UINT64, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<float32_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_FLOAT32, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<float64_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_FLOAT64, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<floatmax_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_FLOATMAX, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<complex128_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_COMPLEX128, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<CSGObject*>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_SGOBJECT, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGString<bool> >* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_STRING, PT_BOOL, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGString<char> >* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_STRING, PT_CHAR, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGString<int8_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_STRING, PT_INT8, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGString<uint8_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_STRING, PT_UINT8, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGString<int16_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_STRING, PT_INT16, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGString<uint16_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_STRING, PT_UINT16, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGString<int32_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_STRING, PT_INT32, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGString<uint32_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_STRING, PT_UINT32, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGString<int64_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_STRING, PT_INT64, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGString<uint64_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_STRING, PT_UINT64, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGString<float32_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_STRING, PT_FLOAT32, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGString<float64_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_STRING, PT_FLOAT64, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGString<floatmax_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_STRING, PT_FLOATMAX, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGSparseVector<bool> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_SPARSE, PT_BOOL, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGSparseVector<char> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_SPARSE, PT_CHAR, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGSparseVector<int8_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_SPARSE, PT_INT8, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGSparseVector<uint8_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_SPARSE, PT_UINT8, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGSparseVector<int16_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_SPARSE, PT_INT16, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGSparseVector<uint16_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_SPARSE, PT_UINT16, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGSparseVector<int32_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_SPARSE, PT_INT32, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGSparseVector<uint32_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_SPARSE, PT_UINT32, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGSparseVector<int64_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_SPARSE, PT_INT64, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGSparseVector<uint64_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_SPARSE, PT_UINT64, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGSparseVector<float32_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_SPARSE, PT_FLOAT32, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGSparseVector<float64_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_SPARSE, PT_FLOAT64, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGSparseVector<floatmax_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_SPARSE, PT_FLOATMAX, &param->vlen);
	add_type(&type, &param->vector, name, description);
}

void Parameter::add(SGVector<SGSparseVector<complex128_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_SPARSE, PT_COMPLEX128, &param->vlen);
	add_type(&type, &param->vector, name, description);
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
	complex128_t** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_COMPLEX128, length_y,
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
Parameter::add_matrix(SGString<bool>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_BOOL, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGString<char>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_CHAR, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGString<int8_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_INT8, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGString<uint8_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_UINT8, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGString<int16_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_INT16, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGString<uint16_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_UINT16, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGString<int32_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_INT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGString<uint32_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_UINT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGString<int64_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_INT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGString<uint64_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_UINT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGString<float32_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_FLOAT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGString<float64_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_FLOAT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGString<floatmax_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_STRING, PT_FLOATMAX, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGSparseVector<bool>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_BOOL, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGSparseVector<char>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_CHAR, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGSparseVector<int8_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_INT8, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGSparseVector<uint8_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_UINT8, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGSparseVector<int16_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_INT16, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGSparseVector<uint16_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_UINT16, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGSparseVector<int32_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_INT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGSparseVector<uint32_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_UINT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGSparseVector<int64_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_INT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGSparseVector<uint64_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_UINT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGSparseVector<float32_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_FLOAT32, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGSparseVector<float64_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_FLOAT64, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGSparseVector<floatmax_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_FLOATMAX, length_y,
					 length_x);
	add_type(&type, param, name, description);
}

void
Parameter::add_matrix(SGSparseVector<complex128_t>** param,
					  index_t* length_y, index_t* length_x,
					  const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_SPARSE, PT_COMPLEX128, length_y,
					 length_x);
	add_type(&type, param, name, description);
}




void Parameter::add(SGMatrix<bool>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_BOOL, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<char>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_CHAR, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<int8_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_INT8, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<uint8_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_UINT8, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<int16_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_INT16, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<uint16_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_UINT16, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<int32_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_INT32, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<uint32_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_UINT32, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<int64_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_INT64, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<uint64_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_UINT64, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<float32_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_FLOAT32, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<float64_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_FLOAT64, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<floatmax_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_FLOATMAX, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<complex128_t>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_COMPLEX128, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<CSGObject*>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_SGOBJECT, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGString<bool> >* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_STRING, PT_BOOL, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGString<char> >* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_STRING, PT_CHAR, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGString<int8_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_STRING, PT_INT8, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGString<uint8_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_STRING, PT_UINT8, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGString<int16_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_STRING, PT_INT16, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGString<uint16_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_STRING, PT_UINT16, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGString<int32_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_STRING, PT_INT32, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGString<uint32_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_STRING, PT_UINT32, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGString<int64_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_STRING, PT_INT64, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGString<uint64_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_STRING, PT_UINT64, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGString<float32_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_STRING, PT_FLOAT32, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGString<float64_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_STRING, PT_FLOAT64, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGString<floatmax_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_STRING, PT_FLOATMAX, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGSparseVector<bool> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_BOOL, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGSparseVector<char> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_CHAR, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGSparseVector<int8_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_INT8, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGSparseVector<uint8_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_UINT8, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGSparseVector<int16_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_INT16, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGSparseVector<uint16_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_UINT16, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGSparseVector<int32_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_INT32, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGSparseVector<uint32_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_UINT32, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGSparseVector<int64_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_INT64, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGSparseVector<uint64_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_UINT64, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGSparseVector<float32_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOAT32, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGSparseVector<float64_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOAT64, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGSparseVector<floatmax_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOATMAX, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGMatrix<SGSparseVector<complex128_t> >* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_COMPLEX128, &param->num_rows,
			&param->num_cols);
	add_type(&type, &param->matrix, name, description);
}

void Parameter::add(SGSparseMatrix<bool>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_BOOL, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

void Parameter::add(SGSparseMatrix<char>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_CHAR, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

void Parameter::add(SGSparseMatrix<int8_t>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_INT8, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

void Parameter::add(SGSparseMatrix<uint8_t>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_UINT8, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

void Parameter::add(SGSparseMatrix<int16_t>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_INT16, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

void Parameter::add(SGSparseMatrix<uint16_t>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_UINT16, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

void Parameter::add(SGSparseMatrix<int32_t>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_INT32, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

void Parameter::add(SGSparseMatrix<uint32_t>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_UINT32, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

void Parameter::add(SGSparseMatrix<int64_t>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_INT64, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

void Parameter::add(SGSparseMatrix<uint64_t>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_UINT64, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

void Parameter::add(SGSparseMatrix<float32_t>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOAT32, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

void Parameter::add(SGSparseMatrix<float64_t>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOAT64, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

void Parameter::add(SGSparseMatrix<floatmax_t>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_FLOATMAX, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

void Parameter::add(SGSparseMatrix<complex128_t>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_COMPLEX128, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

void Parameter::add(SGSparseMatrix<CSGObject*>* param,
		const char* name, const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_SPARSE, PT_SGOBJECT, &param->num_vectors,
			&param->num_features);
	add_type(&type, &param->sparse_matrix, name, description);
}

/* **************************************************************** */
/* End of wrappers  */

TParameter::TParameter(const TSGDataType* datatype, void* parameter,
					   const char* name, const char* description)
	:m_datatype(*datatype)
{
	m_parameter = parameter;
	m_name = get_strdup(name);
	m_description = get_strdup(description);
}

TParameter::~TParameter()
{
	SG_FREE(m_description);
	SG_FREE(m_name);
}

char*
TParameter::new_prefix(const char* s1, const char* s2)
{
	char* tmp = SG_MALLOC(char, strlen(s1)+strlen(s2)+2);

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
TParameter::delete_cont()
{
	if (*(void**) m_parameter != NULL) {
		index_t old_length = m_datatype.m_length_y ? *m_datatype.m_length_y : 0;
		switch (m_datatype.m_ctype) {
		case CT_NDARRAY:
			SG_SNOTIMPLEMENTED
			break;
		case CT_MATRIX: case CT_SGMATRIX:
			old_length *= *m_datatype.m_length_x; break;
		case CT_SCALAR: case CT_VECTOR: case CT_SGVECTOR: break;
		case CT_UNDEFINED: default:
			SG_SERROR("Implementation error: undefined container type\n");
			break;
		}

		switch (m_datatype.m_stype) {
		case ST_NONE:
			switch (m_datatype.m_ptype) {
			case PT_BOOL:
				SG_FREE(*(bool**) m_parameter); break;
			case PT_CHAR:
				SG_FREE(*(char**) m_parameter); break;
			case PT_INT8:
				SG_FREE(*(int8_t**) m_parameter); break;
			case PT_UINT8:
				SG_FREE(*(uint8_t**) m_parameter); break;
			case PT_INT16:
				SG_FREE(*(int16_t**) m_parameter); break;
			case PT_UINT16:
				SG_FREE(*(uint16_t**) m_parameter); break;
			case PT_INT32:
				SG_FREE(*(int32_t**) m_parameter); break;
			case PT_UINT32:
				SG_FREE(*(uint32_t**) m_parameter); break;
			case PT_INT64:
				SG_FREE(*(int64_t**) m_parameter); break;
			case PT_UINT64:
				SG_FREE(*(uint64_t**) m_parameter); break;
			case PT_FLOAT32:
				SG_FREE(*(float32_t**) m_parameter); break;
			case PT_FLOAT64:
				SG_FREE(*(float64_t**) m_parameter); break;
			case PT_FLOATMAX:
				SG_FREE(*(floatmax_t**) m_parameter); break;
			case PT_COMPLEX128:
				SG_FREE(*(complex128_t**) m_parameter); break;
			case PT_SGOBJECT:
			{
				CSGObject** buf = *(CSGObject***) m_parameter;

				for (index_t i=0; i<old_length; i++)
					SG_UNREF(buf[i]);

				SG_FREE(buf);
				break;
			}
			case PT_UNDEFINED: default:
				SG_SERROR("Implementation error: undefined primitive type\n");
				break;
			}
			break;
		case ST_STRING:
		{
			for (index_t i=0; i<old_length; i++) {
				SGString<char>* buf = (SGString<char>*) (*(char**)
						m_parameter + i *m_datatype.sizeof_stype());
				if (buf->slen > 0) SG_FREE(buf->string);
			break;
		}
			}

			switch (m_datatype.m_ptype) {
			case PT_BOOL:
				SG_FREE(*(SGString<bool>**) m_parameter); break;
			case PT_CHAR:
				SG_FREE(*(SGString<char>**) m_parameter); break;
			case PT_INT8:
				SG_FREE(*(SGString<int8_t>**) m_parameter); break;
			case PT_UINT8:
				SG_FREE(*(SGString<uint8_t>**) m_parameter); break;
			case PT_INT16:
				SG_FREE(*(SGString<int16_t>**) m_parameter); break;
			case PT_UINT16:
				SG_FREE(*(SGString<uint16_t>**) m_parameter); break;
			case PT_INT32:
				SG_FREE(*(SGString<int32_t>**) m_parameter); break;
			case PT_UINT32:
				SG_FREE(*(SGString<uint32_t>**) m_parameter); break;
			case PT_INT64:
				SG_FREE(*(SGString<int64_t>**) m_parameter); break;
			case PT_UINT64:
				SG_FREE(*(SGString<uint64_t>**) m_parameter); break;
			case PT_FLOAT32:
				SG_FREE(*(SGString<float32_t>**) m_parameter); break;
			case PT_FLOAT64:
				SG_FREE(*(SGString<float64_t>**) m_parameter); break;
			case PT_FLOATMAX:
				SG_FREE(*(SGString<floatmax_t>**) m_parameter); break;
			case PT_COMPLEX128:
				SG_SERROR("TParameter::delete_cont(): Parameters of strings"
						" of complex128_t are not supported");
				break;
			case PT_SGOBJECT:
				SG_SERROR("TParameter::delete_cont(): Implementation "
						 "error: Could not delete "
						 "String<SGSerializable*>");
				break;
			case PT_UNDEFINED: default:
				SG_SERROR("Implementation error: undefined primitive type\n");
				break;
			}
			break;
		case ST_SPARSE:
			for (index_t i=0; i<old_length; i++) {
				SGSparseVector<char>* buf = (SGSparseVector<char>*) (*(char**)
						m_parameter + i *m_datatype.sizeof_stype());
				if (buf->num_feat_entries > 0) SG_FREE(buf->features);
			}

			switch (m_datatype.m_ptype) {
			case PT_BOOL:
				SG_FREE(*(SGSparseVector<bool>**) m_parameter); break;
			case PT_CHAR:
				SG_FREE(*(SGSparseVector<char>**) m_parameter); break;
			case PT_INT8:
				SG_FREE(*(SGSparseVector<int8_t>**) m_parameter); break;
			case PT_UINT8:
				SG_FREE(*(SGSparseVector<uint8_t>**) m_parameter); break;
			case PT_INT16:
				SG_FREE(*(SGSparseVector<int16_t>**) m_parameter); break;
			case PT_UINT16:
				SG_FREE(*(SGSparseVector<uint16_t>**) m_parameter); break;
			case PT_INT32:
				SG_FREE(*(SGSparseVector<int32_t>**) m_parameter); break;
			case PT_UINT32:
				SG_FREE(*(SGSparseVector<uint32_t>**) m_parameter); break;
			case PT_INT64:
				SG_FREE(*(SGSparseVector<int64_t>**) m_parameter); break;
			case PT_UINT64:
				SG_FREE(*(SGSparseVector<uint64_t>**) m_parameter); break;
			case PT_FLOAT32:
				SG_FREE(*(SGSparseVector<float32_t>**) m_parameter); break;
			case PT_FLOAT64:
				SG_FREE(*(SGSparseVector<float64_t>**) m_parameter); break;
			case PT_FLOATMAX:
				SG_FREE(*(SGSparseVector<floatmax_t>**) m_parameter); break;
			case PT_COMPLEX128:
				SG_FREE(*(SGSparseVector<complex128_t>**) m_parameter); break;
			case PT_SGOBJECT:
				SG_SERROR("TParameter::delete_cont(): Implementation "
						 "error: Could not delete "
						 "Sparse<SGSerializable*>");
				break;
			case PT_UNDEFINED: default:
				SG_SERROR("Implementation error: undefined primitive type\n");
				break;
			}
			break;
		case ST_UNDEFINED: default:
			SG_SERROR("Implementation error: undefined structure type\n");
			break;
		} /* switch (m_datatype.m_stype)  */
	} /* if (*(void**) m_parameter != NULL)  */

	*(void**) m_parameter = NULL;
}

void
TParameter::new_cont(SGVector<index_t> dims)
{
	char* s=SG_MALLOC(char, 200);
	m_datatype.to_string(s, 200);
	SG_SDEBUG("entering TParameter::new_cont for \"%s\" of type %s\n",
			s, m_name ? m_name : "(nil)");
	SG_FREE(s);
	delete_cont();

	index_t new_length = dims.product();
	if (new_length == 0) return;

	switch (m_datatype.m_stype) {
	case ST_NONE:
		switch (m_datatype.m_ptype) {
		case PT_BOOL:
			*(bool**) m_parameter
				= SG_MALLOC(bool, new_length); break;
		case PT_CHAR:
			*(char**) m_parameter
				= SG_MALLOC(char, new_length); break;
		case PT_INT8:
			*(int8_t**) m_parameter
				= SG_MALLOC(int8_t, new_length); break;
		case PT_UINT8:
			*(uint8_t**) m_parameter
				= SG_MALLOC(uint8_t, new_length); break;
		case PT_INT16:
			*(int16_t**) m_parameter
				= SG_MALLOC(int16_t, new_length); break;
		case PT_UINT16:
			*(uint16_t**) m_parameter
				= SG_MALLOC(uint16_t, new_length); break;
		case PT_INT32:
			*(int32_t**) m_parameter
				= SG_MALLOC(int32_t, new_length); break;
		case PT_UINT32:
			*(uint32_t**) m_parameter
				= SG_MALLOC(uint32_t, new_length); break;
		case PT_INT64:
			*(int64_t**) m_parameter
				= SG_MALLOC(int64_t, new_length); break;
		case PT_UINT64:
			*(uint64_t**) m_parameter
				= SG_MALLOC(uint64_t, new_length); break;
		case PT_FLOAT32:
			*(float32_t**) m_parameter
				= SG_MALLOC(float32_t, new_length); break;
		case PT_FLOAT64:
			*(float64_t**) m_parameter
				= SG_MALLOC(float64_t, new_length); break;
		case PT_FLOATMAX:
			*(floatmax_t**) m_parameter
				= SG_MALLOC(floatmax_t, new_length); break;
		case PT_COMPLEX128:
			*(complex128_t**) m_parameter
				= SG_MALLOC(complex128_t, new_length); break;
		case PT_SGOBJECT:
			*(CSGObject***) m_parameter
				= SG_CALLOC(CSGObject*, new_length);
			break;
		case PT_UNDEFINED: default:
			SG_SERROR("Implementation error: undefined primitive type\n");
			break;
		}
		break;
	case ST_STRING:
		switch (m_datatype.m_ptype) {
		case PT_BOOL:
			*(SGString<bool>**) m_parameter
				= SG_MALLOC(SGString<bool>, new_length); break;
		case PT_CHAR:
			*(SGString<char>**) m_parameter
				= SG_MALLOC(SGString<char>, new_length); break;
		case PT_INT8:
			*(SGString<int8_t>**) m_parameter
				= SG_MALLOC(SGString<int8_t>, new_length); break;
		case PT_UINT8:
			*(SGString<uint8_t>**) m_parameter
				= SG_MALLOC(SGString<uint8_t>, new_length); break;
		case PT_INT16:
			*(SGString<int16_t>**) m_parameter
				= SG_MALLOC(SGString<int16_t>, new_length); break;
		case PT_UINT16:
			*(SGString<uint16_t>**) m_parameter
				= SG_MALLOC(SGString<uint16_t>, new_length); break;
		case PT_INT32:
			*(SGString<int32_t>**) m_parameter
				= SG_MALLOC(SGString<int32_t>, new_length); break;
		case PT_UINT32:
			*(SGString<uint32_t>**) m_parameter
				= SG_MALLOC(SGString<uint32_t>, new_length); break;
		case PT_INT64:
			*(SGString<int64_t>**) m_parameter
				= SG_MALLOC(SGString<int64_t>, new_length); break;
		case PT_UINT64:
			*(SGString<uint64_t>**) m_parameter
				= SG_MALLOC(SGString<uint64_t>, new_length); break;
		case PT_FLOAT32:
			*(SGString<float32_t>**) m_parameter
				= SG_MALLOC(SGString<float32_t>, new_length); break;
		case PT_FLOAT64:
			*(SGString<float64_t>**) m_parameter
				= SG_MALLOC(SGString<float64_t>, new_length); break;
		case PT_FLOATMAX:
			*(SGString<floatmax_t>**) m_parameter
				= SG_MALLOC(SGString<floatmax_t>, new_length); break;
		case PT_COMPLEX128:
			SG_SERROR("TParameter::new_cont(): Implementation "
					 "error: Could not allocate "
					 "String<complex128>");
			break;
		case PT_SGOBJECT:
			SG_SERROR("TParameter::new_cont(): Implementation "
					 "error: Could not allocate "
					 "String<SGSerializable*>");
			break;
		case PT_UNDEFINED: default:
			SG_SERROR("Implementation error: undefined primitive type\n");
			break;
		}
		memset(*(void**) m_parameter, 0, new_length
			   *m_datatype.sizeof_stype());
		break;
	case ST_SPARSE:
		switch (m_datatype.m_ptype) {
		case PT_BOOL:
			*(SGSparseVector<bool>**) m_parameter
				= SG_MALLOC(SGSparseVector<bool>, new_length); break;
		case PT_CHAR:
			*(SGSparseVector<char>**) m_parameter
				= SG_MALLOC(SGSparseVector<char>, new_length); break;
		case PT_INT8:
			*(SGSparseVector<int8_t>**) m_parameter
				= SG_MALLOC(SGSparseVector<int8_t>, new_length); break;
		case PT_UINT8:
			*(SGSparseVector<uint8_t>**) m_parameter
				= SG_MALLOC(SGSparseVector<uint8_t>, new_length); break;
		case PT_INT16:
			*(SGSparseVector<int16_t>**) m_parameter
				= SG_MALLOC(SGSparseVector<int16_t>, new_length); break;
		case PT_UINT16:
			*(SGSparseVector<uint16_t>**) m_parameter
				= SG_MALLOC(SGSparseVector<uint16_t>, new_length); break;
		case PT_INT32:
			*(SGSparseVector<int32_t>**) m_parameter
				= SG_MALLOC(SGSparseVector<int32_t>, new_length);
			break;
		case PT_UINT32:
			*(SGSparseVector<uint32_t>**) m_parameter
				= SG_MALLOC(SGSparseVector<uint32_t>, new_length); break;
		case PT_INT64:
			*(SGSparseVector<int64_t>**) m_parameter
				= SG_MALLOC(SGSparseVector<int64_t>, new_length); break;
		case PT_UINT64:
			*(SGSparseVector<uint64_t>**) m_parameter
				= SG_MALLOC(SGSparseVector<uint64_t>, new_length); break;
		case PT_FLOAT32:
			*(SGSparseVector<float32_t>**) m_parameter
				= SG_MALLOC(SGSparseVector<float32_t>, new_length); break;
		case PT_FLOAT64:
			*(SGSparseVector<float64_t>**) m_parameter
				= SG_MALLOC(SGSparseVector<float64_t>, new_length); break;
		case PT_FLOATMAX:
			*(SGSparseVector<floatmax_t>**) m_parameter
				= SG_MALLOC(SGSparseVector<floatmax_t>, new_length); break;
		case PT_COMPLEX128:
			*(SGSparseVector<complex128_t>**) m_parameter
				= SG_MALLOC(SGSparseVector<complex128_t>, new_length); break;
		case PT_SGOBJECT:
			SG_SERROR("TParameter::new_cont(): Implementation "
					 "error: Could not allocate "
					 "Sparse<SGSerializable*>");
			break;
		case PT_UNDEFINED: default:
			SG_SERROR("Implementation error: undefined primitive type\n");
			break;
		}
		break;
	case ST_UNDEFINED: default:
		SG_SERROR("Implementation error: undefined structure type\n");
		break;
	} /* switch (m_datatype.m_stype)  */

	s=SG_MALLOC(char, 200);
	m_datatype.to_string(s, 200);
	SG_SDEBUG("leaving TParameter::new_cont for \"%s\" of type %s\n",
			s, m_name ? m_name : "(nil)");
	SG_FREE(s);
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
	SGString<char>* str_ptr = (SGString<char>*) param;
	SGSparseVector<char>* spr_ptr = (SGSparseVector<char>*) param;
	index_t len_real;

	switch (m_datatype.m_stype) {
	case ST_NONE:
		if (!save_ptype(file, param, prefix)) return false;
		break;
	case ST_STRING:
		len_real = str_ptr->slen;
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
				&m_datatype, m_name, prefix, len_real)) return false;
		for (index_t i=0; i<len_real; i++) {
			SGSparseVectorEntry<char>* cur = (SGSparseVectorEntry<char>*)
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
				&m_datatype, m_name, prefix, len_real)) return false;
		break;
	case ST_UNDEFINED: default:
		SG_SERROR("Implementation error: undefined structure type\n");
		break;
	}

	return true;
}

bool
TParameter::load_stype(CSerializableFile* file, void* param,
					   const char* prefix)
{
	SGString<char>* str_ptr = (SGString<char>*) param;
	SGSparseVector<char>* spr_ptr = (SGSparseVector<char>*) param;
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
			? SG_MALLOC(char, len_real*m_datatype.sizeof_ptype()): NULL;
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
		str_ptr->slen = len_real;
		break;
	case ST_SPARSE:
		if (!file->read_sparse_begin(
				&m_datatype, m_name, prefix, &len_real)) return false;
		spr_ptr->features = len_real > 0? (SGSparseVectorEntry<char>*)
			SG_MALLOC(char, len_real *TSGDataType::sizeof_sparseentry(
				m_datatype.m_ptype)): NULL;
		for (index_t i=0; i<len_real; i++) {
			SGSparseVectorEntry<char>* cur = (SGSparseVectorEntry<char>*)
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

		if (!file->read_sparse_end(&m_datatype, m_name, prefix, len_real))
			return false;

		spr_ptr->num_feat_entries = len_real;
		break;
	case ST_UNDEFINED: default:
		SG_SERROR("Implementation error: undefined structure type\n");
		break;
	}

	return true;
}

void TParameter::get_incremental_hash(
		uint32_t& hash, uint32_t& carry, uint32_t& total_length)
{

	switch (m_datatype.m_ctype)
	{
	case CT_NDARRAY:
		SG_SNOTIMPLEMENTED
		break;
	case CT_SCALAR:
	{
	    uint8_t* data = ((uint8_t*) m_parameter);
		uint32_t size = m_datatype.sizeof_stype();
		total_length += size;
		CHash::IncrementalMurmurHash3(
				&hash, &carry, data, size);
		break;
	}
	case CT_VECTOR: case CT_MATRIX: case CT_SGVECTOR: case CT_SGMATRIX:
	{
		index_t len_real_y = 0, len_real_x = 0;

		if (m_datatype.m_length_y)
			len_real_y = *m_datatype.m_length_y;

		else
			len_real_y = 1;

		if (*(void**) m_parameter == NULL && len_real_y != 0)
		{
			SG_SWARNING("Inconsistency between data structure and "
					"len_y during hashing `%s'!  Continuing with "
					"len_y=0.\n",
					m_name);
			len_real_y = 0;
		}

		switch (m_datatype.m_ctype)
		{
		case CT_NDARRAY:
			SG_SNOTIMPLEMENTED
			break;
		case CT_VECTOR: case CT_SGVECTOR:
			len_real_x = 1;
			break;
		case CT_MATRIX: case CT_SGMATRIX:
			len_real_x = *m_datatype.m_length_x;

			if (*(void**) m_parameter == NULL && len_real_x != 0)
			{
				SG_SWARNING("Inconsistency between data structure and "
						"len_x during hashing %s'!  Continuing "
						"with len_x=0.\n",
						m_name);
				len_real_x = 0;
			}

			if (len_real_x *len_real_y == 0)
				len_real_x = len_real_y = 0;

			break;

		case CT_SCALAR: break;
		case CT_UNDEFINED: default:
			SG_SERROR("Implementation error: undefined container type\n");
			break;
		}
		uint32_t size = (len_real_x*len_real_y)*m_datatype.sizeof_stype();

		total_length += size;

	        uint8_t* data = (*(uint8_t**) m_parameter);

		CHash::IncrementalMurmurHash3(
				&hash, &carry, data, size);
		break;
	}
	case CT_UNDEFINED: default:
		SG_SERROR("Implementation error: undefined container type\n");
		break;
	}
}

bool
TParameter::is_valid()
{
	return m_datatype.get_num_elements() > 0;
}

bool
TParameter::save(CSerializableFile* file, const char* prefix)
{
	const int32_t buflen=100;
	char* buf=SG_MALLOC(char, buflen);
	m_datatype.to_string(buf, buflen);
	SG_SINFO("Saving parameter '%s' of type '%s'\n", m_name, buf)
	SG_FREE(buf);

	if (!file->write_type_begin(&m_datatype, m_name, prefix))
		return false;

	switch (m_datatype.m_ctype) {
	case CT_NDARRAY:
		SG_SNOTIMPLEMENTED
		break;
	case CT_SCALAR:
		if (!save_stype(file, m_parameter, prefix)) return false;
		break;
	case CT_VECTOR: case CT_MATRIX: case CT_SGVECTOR: case CT_SGMATRIX:
	{
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
		case CT_NDARRAY:
			SG_SNOTIMPLEMENTED
			break;
		case CT_VECTOR: case CT_SGVECTOR:
			len_real_x = 1;
			break;
		case CT_MATRIX: case CT_SGMATRIX:
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
		case CT_UNDEFINED: default:
			SG_SERROR("Implementation error: undefined container type\n");
			break;
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
	case CT_UNDEFINED: default:
		SG_SERROR("Implementation error: undefined container type\n");
		break;
	}

	if (!file->write_type_end(&m_datatype, m_name, prefix))
		return false;

	return true;
}

bool
TParameter::load(CSerializableFile* file, const char* prefix)
{
	REQUIRE(file != NULL, "Serializable file object should be != NULL\n");

	const int32_t buflen=100;
	char* buf=SG_MALLOC(char, buflen);
	m_datatype.to_string(buf, buflen);
	SG_SDEBUG("Loading parameter '%s' of type '%s'\n", m_name, buf)
	SG_FREE(buf);

	if (!file->read_type_begin(&m_datatype, m_name, prefix))
		return false;

	switch (m_datatype.m_ctype)
	{
		case CT_NDARRAY:
			SG_SNOTIMPLEMENTED
			break;
		case CT_SCALAR:
			if (!load_stype(file, m_parameter, prefix))
				return false;
			break;

		case CT_VECTOR: case CT_MATRIX: case CT_SGVECTOR: case CT_SGMATRIX:
		{
			SGVector<index_t> dims(2);
			dims.zero();

			if (!file->read_cont_begin(&m_datatype, m_name, prefix,
						&dims.vector[1], &dims.vector[0]))
				return false;

			switch (m_datatype.m_ctype)
			{
				case CT_NDARRAY:
					SG_SNOTIMPLEMENTED
					break;
				case CT_VECTOR: case CT_SGVECTOR:
					dims[0]=1;
					new_cont(dims);
					break;
				case CT_MATRIX: case CT_SGMATRIX:
					new_cont(dims);
					break;
				case CT_SCALAR:
					break;
				case CT_UNDEFINED: default:
					SG_SERROR("Implementation error: undefined container type\n");
					break;
			}

			for (index_t x=0; x<dims[0]; x++)
			{
				for (index_t y=0; y<dims[1]; y++)
				{
					if (!file->read_item_begin(
								&m_datatype, m_name, prefix, y, x))
						return false;

					if (!load_stype(
								file, (*(char**) m_parameter)
								+ (x*dims[1] + y)*m_datatype.sizeof_stype(),
								prefix)) return false;
					if (!file->read_item_end(
								&m_datatype, m_name, prefix, y, x))
						return false;
				}
			}

			switch (m_datatype.m_ctype)
			{
				case CT_NDARRAY:
					SG_SNOTIMPLEMENTED
					break;
				case CT_VECTOR: case CT_SGVECTOR:
					*m_datatype.m_length_y = dims[1];
					break;
				case CT_MATRIX: case CT_SGMATRIX:
					*m_datatype.m_length_y = dims[1];
					*m_datatype.m_length_x = dims[0];
					break;
				case CT_SCALAR:
					break;
				case CT_UNDEFINED: default:
					SG_SERROR("Implementation error: undefined container type\n");
					break;
			}

			if (!file->read_cont_end(&m_datatype, m_name, prefix,
						dims[1], dims[0]))
				return false;

			break;
		}
		case CT_UNDEFINED: default:
			SG_SERROR("Implementation error: undefined container type\n");
			break;
	}

	if (!file->read_type_end(&m_datatype, m_name, prefix))
		return false;

	return true;
}

/*
  Initializing m_params(1) with small preallocation-size, because Parameter
  will be constructed several times for EACH SGObject instance.
 */
Parameter::Parameter() : m_params(1)
{
	SG_REF(sg_io);
}

Parameter::~Parameter()
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
		SG_SERROR("FATAL: Parameter::add_type(): `name' is empty!\n")

	for (size_t i=0; i<strlen(name); ++i)
	{
		if (!std::isalnum(name[i]) && name[i]!='_' && name[i]!='.')
		{
			SG_SERROR("Character %d of parameter with name \"%s\" is illegal "
					"(only alnum or underscore is allowed)\n",
					i, name);
		}
	}

	for (int32_t i=0; i<get_num_parameters(); i++)
		if (strcmp(m_params.get_element(i)->m_name, name) == 0)
			SG_SERROR("FATAL: Parameter::add_type(): "
					 "Double parameter `%s'!\n", name);

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

void Parameter::set_from_parameters(Parameter* params)
{
	/* iterate over parameters in the given list */
	for (index_t i=0; i<params->get_num_parameters(); ++i)
	{
		TParameter* current=params->get_parameter(i);
		TSGDataType current_type=current->m_datatype;

		ASSERT(m_params.get_num_elements())

		/* search for own parameter with same name and check types if found */
		TParameter* own=NULL;
		for (index_t j=0; j<m_params.get_num_elements(); ++j)
		{
			own=m_params.get_element(j);
			if (!strcmp(own->m_name, current->m_name))
			{
				if (own->m_datatype==current_type)
				{
					own=m_params.get_element(j);
					break;
				}
				else
				{
					index_t l=200;
					char* given_type=SG_MALLOC(char, l);
					char* own_type=SG_MALLOC(char, l);
					current->m_datatype.to_string(given_type, l);
					own->m_datatype.to_string(own_type, l);
					SG_SERROR("given parameter \"%s\" has a different type (%s)"
							" than existing one (%s)\n", current->m_name,
							given_type, own_type);
					SG_FREE(given_type);
					SG_FREE(own_type);
				}
			}
			else
				own=NULL;
		}

		if (!own)
		{
			SG_SERROR("parameter with name %s does not exist\n",
					current->m_name);
		}

		/* check if parameter contained CSGobjects (update reference counts) */
		if (current_type.m_ptype==PT_SGOBJECT)
		{
			/* PT_SGOBJECT only occurs for ST_NONE */
			if (own->m_datatype.m_stype==ST_NONE)
			{
				if (own->m_datatype.m_ctype==CT_SCALAR)
				{
					CSGObject** to_unref=(CSGObject**) own->m_parameter;
					CSGObject** to_ref=(CSGObject**) current->m_parameter;

					if ((*to_ref)!=(*to_unref))
					{
						SG_REF((*to_ref));
						SG_UNREF((*to_unref));
					}

				}
				else
				{
					/* unref all SGObjects and reference the new ones */
					CSGObject*** to_unref=(CSGObject***) own->m_parameter;
					CSGObject*** to_ref=(CSGObject***) current->m_parameter;

					for (index_t j=0; j<own->m_datatype.get_num_elements(); ++j)
					{
						if ((*to_ref)[j]!=(*to_unref)[j])
						{
							SG_REF(((*to_ref)[j]));
							SG_UNREF(((*to_unref)[j]));
						}
					}
				}
			}
			else
				SG_SERROR("primitive type PT_SGOBJECT occurred with structure "
						"type other than ST_NONE");
		}

		/* construct pointers to the to be copied parameter data */
		void* dest=NULL;
		void* source=NULL;
		if (current_type.m_ctype==CT_SCALAR)
		{
			/* for scalar values, just copy content the pointer points to */
			dest=own->m_parameter;
			source=current->m_parameter;

			/* in case of CSGObject, pointers are not equal if CSGObjects are
			 * equal, so check. For other values, the pointers are equal and
			 * the not-copying is handled below before the memcpy call */
			if (own->m_datatype.m_ptype==PT_SGOBJECT)
			{
				if (*((CSGObject**)dest) == *((CSGObject**)source))
				{
					dest=NULL;
					source=NULL;
				}
			}
		}
		else
		{
			/* for matrices and vectors, sadly m_parameter has to be
			 * de-referenced once, because a pointer to the array address is
			 * saved, but the array address itself has to be copied.
			 * consequently, for dereferencing, a type distinction is needed */
			switch (own->m_datatype.m_ptype)
			{
			case PT_FLOAT64:
				dest=*((float64_t**) own->m_parameter);
				source=*((float64_t**) current->m_parameter);
				break;
			case PT_SGOBJECT:
				dest=*((CSGObject**) own->m_parameter);
				source=*((CSGObject**) current->m_parameter);
				break;
			default:
				SG_SNOTIMPLEMENTED
				break;
			}
		}

		/* copy parameter data, size in memory is equal because of same type */
		if (dest!=source)
			memcpy(dest, source, own->m_datatype.get_size());
	}
}

void Parameter::add_parameters(Parameter* params)
{
	for (index_t i=0; i<params->get_num_parameters(); ++i)
	{
		TParameter* current=params->get_parameter(i);
		add_type(&(current->m_datatype), current->m_parameter, current->m_name,
				current->m_description);
	}
}

bool Parameter::contains_parameter(const char* name)
{
	for (index_t i=0; i<m_params.get_num_elements(); ++i)
	{
		if (!strcmp(name, m_params[i]->m_name))
			return true;
	}

	return false;
}

bool TParameter::operator==(const TParameter& other) const
{
	bool result=true;
	result&=!strcmp(m_name, other.m_name);
	return result;
}

bool TParameter::operator<(const TParameter& other) const
{
	return strcmp(m_name, other.m_name)<0;
}

bool TParameter::operator>(const TParameter& other) const
{
	return strcmp(m_name, other.m_name)>0;
}

bool TParameter::equals(TParameter* other, float64_t accuracy, bool tolerant)
{
	SG_SDEBUG("entering TParameter::equals()\n");

	if (!other)
	{
		SG_SDEBUG("leaving TParameter::equals(): other parameter is NULL\n");
		return false;
	}

	if (strcmp(m_name, other->m_name))
	{
		SG_SDEBUG("leaving TParameter::equals(): name \"%s\" is different from"
				" other parameter's name \"%s\"\n", m_name, other->m_name);
		return false;
	}

	SG_SDEBUG("Comparing datatypes\n");
	if (!(m_datatype.equals(other->m_datatype)))
	{
		SG_SDEBUG("leaving TParameter::equals(): type of \"%s\" is different "
				"from other parameter's \"%s\" type\n", m_name, other->m_name);
		return false;
	}

	/* avoid comparing NULL */
	if (!m_parameter && !other->m_parameter)
	{
		SG_SDEBUG("leaving TParameter::equals(): both parameters are NULL\n");
		return true;
	}

	if ((!m_parameter && other->m_parameter) || (m_parameter && !other->m_parameter))
	{
		SG_SDEBUG("leaving TParameter::equals(): param1 is at %p while "
				"param2 is at %p\n", m_parameter, other->m_parameter);
		return false;
	}

	SG_SDEBUG("Comparing ctype\n");
	switch (m_datatype.m_ctype)
	{
		case CT_SCALAR:
		{
			SG_SDEBUG("CT_SCALAR\n");
			if (!TParameter::compare_stype(m_datatype.m_stype,
					m_datatype.m_ptype, m_parameter,
					other->m_parameter,
					accuracy, tolerant))
			{
				SG_SDEBUG("leaving TParameter::equals(): scalar data differs\n");
				return false;
			}
			break;
		}
		case CT_VECTOR: case CT_SGVECTOR:
		{
			SG_SDEBUG("CT_VECTOR or CT_SGVECTOR\n");

			/* x is number of processed bytes */
			index_t x=0;
			SG_SDEBUG("length_y: %d\n", *m_datatype.m_length_y)
			for (index_t i=0; i<*m_datatype.m_length_y; ++i)
			{
				SG_SDEBUG("comparing element %d which is %d bytes from start\n",
						i, x);

				void* pointer_a=&((*(char**)m_parameter)[x]);
				void* pointer_b=&((*(char**)other->m_parameter)[x]);

				if (!TParameter::compare_stype(m_datatype.m_stype,
						m_datatype.m_ptype, pointer_a, pointer_b,
						accuracy, tolerant))
				{
					SG_SDEBUG("leaving TParameter::equals(): vector element "
							"differs\n");
					return false;
				}

				x=x+(m_datatype.sizeof_stype());
			}

			break;
		}
		case CT_MATRIX: case CT_SGMATRIX:
		{
			SG_SDEBUG("CT_MATRIX or CT_SGMATRIX\n");

			/* x is number of processed bytes */
			index_t x=0;
			SG_SDEBUG("length_y: %d\n", *m_datatype.m_length_y)
			SG_SDEBUG("length_x: %d\n", *m_datatype.m_length_x)
			int64_t length=0;

			/* For ST_SPARSE, we just need to loop over the rows and compare_stype
			 * does the comparison for one whole row vector at once. For ST_NONE,
			 * however, we need to loop over all elements.
			 */
			if (m_datatype.m_stype==ST_SPARSE)
				length=(*m_datatype.m_length_y);
			else
				length=(*m_datatype.m_length_y) * (*m_datatype.m_length_x);

			for (index_t i=0; i<length; ++i)
			{
				SG_SDEBUG("comparing element %d which is %d bytes from start\n",
						i, x);

				void* pointer_a=&((*(char**)m_parameter)[x]);
				void* pointer_b=&((*(char**)other->m_parameter)[x]);

				if (!TParameter::compare_stype(m_datatype.m_stype,
						m_datatype.m_ptype, pointer_a, pointer_b,
						accuracy, tolerant))
				{
					SG_SDEBUG("leaving TParameter::equals(): vector element "
							"differs\n");
					return false;
				}

				/* For ST_SPARSE, the iteration is on the pointer of SGSparseVectors */
				if (m_datatype.m_stype==ST_SPARSE)
					x=x+(m_datatype.sizeof_stype());
				else
					x=x+(m_datatype.sizeof_stype());
			}

			break;
		}
		case CT_NDARRAY:
		{
			SG_SDEBUG("CT_NDARRAY\n");
			SG_SERROR("TParameter::equals(): Not yet implemented for "
					"CT_NDARRAY!\n");
			break;
		}
		case CT_UNDEFINED: default:
			SG_SERROR("Implementation error: undefined container type\n");
			break;
	}

	SG_SDEBUG("leaving TParameter::equals(): Parameters are equal\n");
	return true;
}

bool TParameter::compare_ptype(EPrimitiveType ptype, void* data1, void* data2,
			float64_t accuracy, bool tolerant)
{
	SG_SDEBUG("entering TParameter::compare_ptype()\n");

	if ((data1 && !data2) || (!data1 && data2))
	{
		SG_SINFO("leaving TParameter::compare_ptype(): data1 is at %p while "
				"data2 is at %p\n", data1, data2);
		return false;
	}

	/** ensure that no NULL data are de-referenced */
	if (!data1 && !data2)
	{
		SG_SDEBUG("leaving TParameter::compare_ptype(): both data are NULL\n");
		return true;
	}

	switch (ptype)
	{
	case PT_BOOL:
	{
		bool casted1=*((bool*)data1);
		bool casted2=*((bool*)data2);

		if (CMath::abs(casted1-casted2)>accuracy)
		{
			SG_SINFO("leaving TParameter::compare_ptype(): PT_BOOL: "
					"data1=%d, data2=%d\n", casted1, casted2);
			return false;
		}
		break;
	}
	case PT_CHAR:
	{
		char casted1=*((char*)data1);
		char casted2=*((char*)data2);

		if (CMath::abs(casted1-casted2)>accuracy)
		{
			SG_SINFO("leaving TParameter::compare_ptype(): PT_CHAR: "
					"data1=%c, data2=%c\n", casted1, casted2);
			return false;
		}
		break;
	}
	case PT_INT8:
	{
		int8_t casted1=*((int8_t*)data1);
		int8_t casted2=*((int8_t*)data2);

		if (CMath::abs(casted1-casted2)>accuracy)
		{
			SG_SINFO("leaving TParameter::compare_ptype(): PT_INT8: "
					"data1=%d, data2=%d\n", casted1, casted2);
			return false;
		}
		break;
	}
	case PT_UINT8:
	{
		uint8_t casted1=*((uint8_t*)data1);
		uint8_t casted2=*((uint8_t*)data2);

		if (CMath::abs(casted1-casted2)>accuracy)
		{
			SG_SINFO("leaving TParameter::compare_ptype(): PT_UINT8: "
					"data1=%d, data2=%d\n", casted1, casted2);
			return false;
		}
		break;
	}
	case PT_INT16:
	{
		int16_t casted1=*((int16_t*)data1);
		int16_t casted2=*((int16_t*)data2);

		if (CMath::abs(casted1-casted2)>accuracy)
		{
			SG_SINFO("leaving TParameter::compare_ptype(): PT_INT16: "
					"data1=%d, data2=%d\n", casted1, casted2);
			return false;
		}
		break;
	}
	case PT_UINT16:
	{
		uint16_t casted1=*((uint16_t*)data1);
		uint16_t casted2=*((uint16_t*)data2);

		if (CMath::abs(casted1-casted2)>accuracy)
		{
			SG_SINFO("leaving TParameter::compare_ptype(): PT_UINT16: "
					"data1=%d, data2=%d\n", casted1, casted2);
			return false;
		}
		break;
	}
	case PT_INT32:
	{
		int32_t casted1=*((int32_t*)data1);
		int32_t casted2=*((int32_t*)data2);

		if (CMath::abs(casted1-casted2)>accuracy)
		{
			SG_SINFO("leaving TParameter::compare_ptype(): PT_INT32: "
					"data1=%d, data2=%d\n", casted1, casted2);
			return false;
		}
		break;
	}
	case PT_UINT32:
	{
		uint32_t casted1=*((uint32_t*)data1);
		uint32_t casted2=*((uint32_t*)data2);

		if (CMath::abs(casted1-casted2)>accuracy)
		{
			SG_SINFO("leaving TParameter::compare_ptype(): PT_UINT32: "
					"data1=%d, data2=%d\n", casted1, casted2);
			return false;
		}
		break;
	}
	case PT_INT64:
	{
		int64_t casted1=*((int64_t*)data1);
		int64_t casted2=*((int64_t*)data2);

		if (CMath::abs(casted1-casted2)>accuracy)
		{
			SG_SINFO("leaving TParameter::compare_ptype(): PT_INT64: "
					"data1=%d, data2=%d\n", casted1, casted2);
			return false;
		}
		break;
	}
	case PT_UINT64:
	{
		uint64_t casted1=*((uint64_t*)data1);
		uint64_t casted2=*((uint64_t*)data2);

		if (CMath::abs(casted1-casted2)>accuracy)
		{
			SG_SINFO("leaving TParameter::compare_ptype(): PT_UINT64: "
					"data1=%d, data2=%d\n", casted1, casted2);
			return false;
		}
		break;
	}
	case PT_FLOAT32:
	{
		float32_t casted1=*((float32_t*)data1);
		float32_t casted2=*((float32_t*)data2);

		SG_SINFO("leaving TParameter::compare_ptype(): PT_FLOAT32: "
				"data1=%f, data2=%f\n", casted1, casted2);

		return CMath::fequals<float32_t>(casted1, casted2, accuracy, tolerant);
		break;
	}
	case PT_FLOAT64:
	{
		float64_t casted1=*((float64_t*)data1);
		float64_t casted2=*((float64_t*)data2);

		SG_SINFO("leaving TParameter::compare_ptype(): PT_FLOAT64: "
				"data1=%f, data2=%f\n", casted1, casted2);

		return CMath::fequals<float64_t>(casted1, casted2, accuracy, tolerant);
		break;
	}
	case PT_FLOATMAX:
	{
		floatmax_t casted1=*((floatmax_t*)data1);
		floatmax_t casted2=*((floatmax_t*)data2);

		SG_SINFO("leaving TParameter::compare_ptype(): PT_FLOATMAX: "
				"data1=%f, data2=%f\n", casted1, casted2);

		return CMath::fequals<floatmax_t>(casted1, casted2, accuracy, tolerant);
		break;
	}
	case PT_COMPLEX128:
	{
		float64_t casted1_real=((complex128_t*)data1)->real();
		float64_t casted1_imag=((complex128_t*)data1)->imag();
		float64_t casted2_real=((complex128_t*)data2)->real();
		float64_t casted2_imag=((complex128_t*)data2)->imag();
		if (CMath::abs(casted1_real-casted2_real)>accuracy ||
			CMath::abs(casted1_imag-casted2_imag)>accuracy)
		{
			SG_SINFO("leaving TParameter::compare_ptype(): PT_COMPLEX128: "
					"data1=%f+i%f, data2=%f+i%f\n",
					casted1_real, casted1_imag,
					casted2_real, casted2_imag);
			return false;
		}
		break;
	}
	case PT_SGOBJECT:
	{
		CSGObject* casted1=*((CSGObject**)data1);
		CSGObject* casted2=*((CSGObject**)data2);

		/* important not to call methods on NULL */
		if (!casted1 && ! casted2)
		{
			SG_SDEBUG("leaving TParameter::compare_ptype(): SGObjects are equal\n");
			return true;
		}

		/* make sure to not call NULL methods */
		if (casted1)
		{
			if (!(casted1->equals(casted2, accuracy)))
			{
				SG_SINFO("leaving TParameter::compare_ptype(): PT_SGOBJECT "
						"equals returned false\n");
				return false;
			}
		}
		else
		{
			if (!(casted2->equals(casted1, accuracy)))
			{
				SG_SINFO("leaving TParameter::compare_ptype(): PT_SGOBJECT "
						"equals returned false\n");
				return false;
			}

		}
		break;
	}
	default:
		SG_SERROR("TParameter::compare_ptype(): Encountered unknown primitive"
				"-type: %d\n", ptype);
		break;
	}

	SG_SDEBUG("leaving TParameter::compare_ptype(): Data are equal\n");
	return true;
}

bool TParameter::copy_ptype(EPrimitiveType ptype, void* source, void* target)
{
	SG_SDEBUG("entering TParameter::copy_ptype()\n");

	/* rather than using memcpy, use the cumbersome way here and cast all types.
	 * This makes it so much easier to debug code.
	 * Copy full stype if this is too slow */
	switch (ptype)
	{
	case PT_BOOL:
	{
		*((bool*)target)=*((bool*)source);
		SG_SDEBUG("after copy of ptype PT_BOOL: source %d, target %d\n",
				*((bool*)source), *((bool*)target));
		break;
	}
	case PT_CHAR:
	{
		*((char*)target)=*((char*)source);
		SG_SDEBUG("after copy of ptype PT_CHAR: source %c, target %c\n",
				*((char*)source), *((char*)target));
		break;
	}
	case PT_INT8:
	{
		*((int8_t*)target)=*((int8_t*)source);
		SG_SDEBUG("after copy of ptype PT_INT8: source %d, target %d\n",
				*((int8_t*)source), *((int8_t*)target));
		break;
	}
	case PT_UINT8:
	{
		*((uint8_t*)target)=*((uint8_t*)source);
		SG_SDEBUG("after copy of ptype PT_UINT8: source %d, target %d\n",
				*((uint8_t*)source), *((uint8_t*)target));
		break;
	}
	case PT_INT16:
	{
		*((int16_t*)target)=*((int16_t*)source);
		SG_SDEBUG("after copy of ptype PT_INT16: source %d, target %d\n",
				*((int16_t*)source), *((int16_t*)target));
		break;
	}
	case PT_UINT16:
	{
		*((uint16_t*)target)=*((uint16_t*)source);
		SG_SDEBUG("after copy of ptype PT_UINT16: source %d, target %d\n",
				*((uint16_t*)source), *((uint16_t*)target));
		break;
	}
	case PT_INT32:
	{
		*((int32_t*)target)=*((int32_t*)source);
		SG_SDEBUG("after copy of ptype PT_INT32: source %d, target %d\n",
				*((int32_t*)source), *((int32_t*)target));
		break;
	}
	case PT_UINT32:
	{
		*((uint32_t*)target)=*((uint32_t*)source);
		SG_SDEBUG("after copy of ptype PT_UINT32: source %d, target %d\n",
				*((uint32_t*)source), *((uint32_t*)target));
		break;
	}
	case PT_INT64:
	{
		*((int64_t*)target)=*((int64_t*)source);
		SG_SDEBUG("after copy of ptype PT_INT64: source %d, target %d\n",
				*((int64_t*)source), *((int64_t*)target));
		break;
	}
	case PT_UINT64:
	{
		*((uint64_t*)target)=*((uint64_t*)source);
		SG_SDEBUG("after copy of ptype PT_UINT64: source %d, target %d\n",
				*((uint64_t*)source), *((uint64_t*)target));
		break;
	}
	case PT_FLOAT32:
	{
		*((float32_t*)target)=*((float32_t*)source);
		SG_SDEBUG("after copy of ptype PT_FLOAT32: source %f, target %f\n",
				*((float32_t*)source), *((float32_t*)target));
		break;
	}
	case PT_FLOAT64:
	{
		*((float64_t*)target)=*((float64_t*)source);
		SG_SDEBUG("after copy of ptype PT_FLOAT64: source %f, target %f\n",
				*((float64_t*)source), *((float64_t*)target));
		break;
	}
	case PT_FLOATMAX:
	{
		*((floatmax_t*)target)=*((floatmax_t*)source);
		SG_SDEBUG("after copy of ptype PT_FLOATMAX: source %Lf, target %Lf\n",
				*((floatmax_t*)source), *((floatmax_t*)target));
		break;
	}
	case PT_COMPLEX128:
	{
		*((complex128_t*)target)=*((complex128_t*)source);
		SG_SDEBUG("after copy of ptype PT_COMPLEX128: "
				"source real %f, target real %f,"
				"source imag %f, target imag %f,"
				"\n",
				((complex128_t*)source)->real(), ((complex128_t*)target)->real(),
				((complex128_t*)source)->imag(), ((complex128_t*)target)->imag());
		break;
	}
	case PT_SGOBJECT:
	{
		CSGObject* casted1=*((CSGObject**)source);
		CSGObject* casted2=*((CSGObject**)target);

		/* important not to call methods on NULL */
		if (!casted1 && ! casted2)
		{
			SG_SDEBUG("leaving TParameter::copy_ptype(): Both SGObjects are NULL\n");
			return true;
		}

		/* make sure to not call NULL methods */
		if (casted1)
		{
			/* in case of overwriting old objects */
			SG_UNREF(*((CSGObject**)target));
			*((CSGObject**)target) = casted1->clone();
		}

		break;
	}
	default:
		SG_SERROR("TParameter::compare_ptype(): Encountered unknown primitive"
				"-type: %d\n", ptype);
		return false;
		break;
	}

	SG_SDEBUG("leaving TParameter::copy_ptype(): Copy successful\n");
	return true;
}

bool TParameter::compare_stype(EStructType stype, EPrimitiveType ptype,
		void* data1, void* data2, float64_t accuracy, bool tolerant)
{
	SG_SDEBUG("entering TParameter::compare_stype()\n");

	size_t size_ptype=TSGDataType::sizeof_ptype(ptype);

	/* Avoid comparing NULL */
	if (!data1 && !data2)
	{
		SG_SDEBUG("leaving TParameter::compare_stype(): both data are NULL\n");
		return true;
	}

	/* If one is NULL, data are not equal */
	if ((data1 && !data2) || (!data1 && data2))
	{
		SG_SINFO("leaving TParameter::compare_stype(): data1 is at %p while "
				"data2 is at %p\n", data1, data2);
		return false;
	}

	switch (stype)
	{
		case ST_NONE:
		{
			SG_SDEBUG("ST_NONE\n");
			return TParameter::compare_ptype(ptype, data1, data2, accuracy, tolerant);
			break;
		}
		case ST_SPARSE:
		{
			SG_SDEBUG("ST_SPARSE\n");
			SGSparseVector<char>* spr_ptr1 = (SGSparseVector<char>*) data1;
			SGSparseVector<char>* spr_ptr2 = (SGSparseVector<char>*) data2;

			if (spr_ptr1->num_feat_entries != spr_ptr2->num_feat_entries)
			{
				SG_SINFO("leaving TParameter::compare_stype(): Length of "
						"sparse vector1 (%d)  is different of vector 2 (%d)\n",
						spr_ptr1->num_feat_entries, spr_ptr2->num_feat_entries);
				return false;
			}

			SG_SDEBUG("Comparing sparse vectors\n");
			for (index_t i=0; i<spr_ptr1->num_feat_entries; ++i)
			{
				SG_SDEBUG("Comparing sparse entry %d at offset %d\n", i,
						i*TSGDataType::sizeof_sparseentry(ptype));

				SGSparseVectorEntry<char>* cur1 = (SGSparseVectorEntry<char>*)
								((char*) spr_ptr1->features + i*TSGDataType
								 ::sizeof_sparseentry(ptype));
				SGSparseVectorEntry<char>* cur2 = (SGSparseVectorEntry<char>*)
								((char*) spr_ptr2->features + i*TSGDataType
								 ::sizeof_sparseentry(ptype));

				/* sparse entries have an offset of the enty pointer depending
				 * on type. Since I cast everything down to char, I need to remove
				 * the char offset and add the offset of the ptype */
				index_t char_offset=TSGDataType::offset_sparseentry(PT_CHAR);
				index_t ptype_offset=TSGDataType::offset_sparseentry(ptype);
				void* pointer1=&(cur1->entry)-char_offset+ptype_offset;
				void* pointer2=&(cur2->entry)-char_offset+ptype_offset;

				if (!TParameter::compare_ptype(ptype, pointer1,
						pointer2, accuracy, tolerant))
				{
					SG_SINFO("leaving TParameter::compare_stype(): Data of"
							" sparse vector element is different\n");
					return false;
				}

				/* also compare feature indices */
				if (cur2->feat_index!=cur1->feat_index)
				{
					SG_SINFO("leaving TParameter::compare_stype(): Feature "
							"index of sparse vector element is different. "
							"source: %d, target: %d\n",
							cur1->feat_index, cur2->feat_index);
					return false;
				}
			}
			break;
		}
		case ST_STRING:
		{
			SG_SDEBUG("ST_STRING\n");
			SGString<char>* str_ptr1 = (SGString<char>*) data1;
			SGString<char>* str_ptr2 = (SGString<char>*) data2;

			if (str_ptr1->slen != str_ptr2->slen)
			{
				SG_SINFO("leaving TParameter::compare_stype(): Length of "
						"string1 (%d)  is different of string2 (%d)\n",
						str_ptr1->slen, str_ptr2->slen);
				return false;
			}

			SG_SDEBUG("Comparing strings\n");
			for (index_t i=0; i<str_ptr1->slen; ++i)
			{
				SG_SDEBUG("Comparing string element %d at offset %d\n", i,
						i*size_ptype);
				void* pointer1=str_ptr1->string+i*size_ptype;
				void* pointer2=str_ptr2->string+i*size_ptype;

				if (!TParameter::compare_ptype(ptype, pointer1,
						pointer2, accuracy, tolerant))
				{
					SG_SINFO("leaving TParameter::compare_stype(): Data of"
							" string element is different\n");
					return false;
				}
			}
			break;
		}
		default:
		{
			SG_SERROR("TParameter::compare_stype(): Undefined struct type\n");
			break;
		}
	}

	SG_SDEBUG("leaving TParameter::compare_stype(): Data were equal\n");
	return true;
}

bool TParameter::copy_stype(EStructType stype, EPrimitiveType ptype,
		void* source, void* target)
{
	SG_SDEBUG("entering TParameter::copy_stype()\n");
	size_t size_ptype=TSGDataType::sizeof_ptype(ptype);

	/* Heiko Strathmann: While I know that copying the stypes string and sparse
	 * element wise is slower than doing the full things, it is way easier to
	 * program and to debug since I already made sure that copy_ptype works as
	 * intended. In addition, strings and vectors of SGObjects can be treated
	 * recursively this way (we dont have cases for this currently, June 2013,
	 * but they can be added without having to modify this code)
	 *
	 * Therefore, this code is very close to the the equals code for
	 * stypes. If it turns out to be too slow (which I doubt), stypes can be
	 * copied with memcpy over the full memory blocks */

	switch (stype)
	{
		case ST_NONE:
		{
			SG_SDEBUG("ST_NONE\n");
			return TParameter::copy_ptype(ptype, source, target);
			break;
		}
		case ST_STRING:
		{
			SG_SDEBUG("ST_STRING\n");
			SGString<char>* source_ptr = (SGString<char>*) source;
			SGString<char>* target_ptr = (SGString<char>*) target;

			if (source_ptr->slen != target_ptr->slen)
			{
				SG_SDEBUG("string lengths different (source: %d vs target: %d),"
						" freeing memory.\n", source_ptr->slen, target_ptr->slen);

				/* if string have different lengths, free data and make equal */
				SG_FREE(target_ptr->string);
				target_ptr->string=NULL;
				target_ptr->slen=0;
			}

			if (!target_ptr->string)
			{
				/* allocate memory if data is NULL */
				size_t num_bytes=source_ptr->slen * size_ptype;

				SG_SDEBUG("target string data NULL, allocating %d bytes.\n",
						num_bytes);
				target_ptr->string=SG_MALLOC(char, num_bytes);
				target_ptr->slen=source_ptr->slen;
			}

			SG_SDEBUG("Copying strings\n");
			for (index_t i=0; i<source_ptr->slen; ++i)
			{
				SG_SDEBUG("Copying string element %d at offset %d\n", i,
						i*size_ptype);
				void* pointer1=source_ptr->string+i*size_ptype;
				void* pointer2=target_ptr->string+i*size_ptype;

				if (!TParameter::copy_ptype(ptype, pointer1, pointer2))
				{
					SG_SDEBUG("leaving TParameter::copy_stype(): Copy of string"
							" element failed.\n");
					return false;
				}
			}
			break;
		}
		case ST_SPARSE:
		{
			SG_SDEBUG("ST_SPARSE\n");
			SGSparseVector<char>* source_ptr = (SGSparseVector<char>*) source;
			SGSparseVector<char>* target_ptr = (SGSparseVector<char>*) target;

			if (source_ptr->num_feat_entries != target_ptr->num_feat_entries)
			{
				SG_SDEBUG("sparse vector lengths different (source: %d vs target: %d),"
						" freeing memory.\n",
						source_ptr->num_feat_entries, target_ptr->num_feat_entries);

				/* if string have different lengths, free data and make equal */
				SG_FREE(target_ptr->features);
				target_ptr->features=NULL;
				target_ptr->num_feat_entries=0;
			}

			if (!target_ptr->features)
			{
				/* allocate memory if data is NULL */
				size_t num_bytes=source_ptr->num_feat_entries *
						TSGDataType::sizeof_sparseentry(ptype);

				SG_SDEBUG("target sparse data NULL, allocating %d bytes.\n",
						num_bytes);
				target_ptr->features=(SGSparseVectorEntry<char>*)SG_MALLOC(char, num_bytes);
				target_ptr->num_feat_entries=source_ptr->num_feat_entries;
			}

			SG_SDEBUG("Copying sparse vectors\n");
			for (index_t i=0; i<source_ptr->num_feat_entries; ++i)
			{
				SG_SDEBUG("Copying sparse entry %d at offset %d\n", i,
						i*TSGDataType::sizeof_sparseentry(ptype));

				SGSparseVectorEntry<char>* cur1 = (SGSparseVectorEntry<char>*)
								((char*) source_ptr->features + i*TSGDataType
										 ::sizeof_sparseentry(ptype));
				SGSparseVectorEntry<char>* cur2 = (SGSparseVectorEntry<char>*)
								((char*) target_ptr->features + i*TSGDataType
										 ::sizeof_sparseentry(ptype));

				/* sparse entries have an offset of the enty pointer depending
				 * on type. Since I cast everything down to char, I need to remove
				 * the char offset and add the offset of the ptype */
				index_t char_offset=TSGDataType::offset_sparseentry(PT_CHAR);
				index_t ptype_offset=TSGDataType::offset_sparseentry(ptype);
				void* pointer1=&(cur1->entry)-char_offset+ptype_offset;
				void* pointer2=&(cur2->entry)-char_offset+ptype_offset;

				if (!TParameter::copy_ptype(ptype, pointer1, pointer2))
				{
					SG_SDEBUG("leaving TParameter::copy_stype(): Copy of sparse"
							" vector element failed\n");
					return false;
				}

				/* afterwards, copy feature indices, wich are the data before
				 * the avove offeet */
				cur2->feat_index=cur1->feat_index;
			}
			break;
		}
		default:
		{
			SG_SERROR("TParameter::copy_stype(): Undefined struct type\n");
			return false;
			break;
		}
	}

	SG_SDEBUG("leaving TParameter::copy_stype(): Copy successful\n");
	return true;
}

bool TParameter::copy(TParameter* target)
{
	SG_SDEBUG("entering TParameter::copy()\n");

	if (!target)
	{
		SG_SDEBUG("leaving TParameter::copy(): other parameter is NULL\n");
		return false;
	}

	if (!m_parameter)
	{
		SG_SDEBUG("leaving TParameter::copy(): m_parameter of source is NULL\n");
		return false;
	}

	if (!target->m_parameter)
	{
		SG_SDEBUG("leaving TParameter::copy(): m_parameter of target is NULL\n");
		return false;
	}

	if (strcmp(m_name, target->m_name))
	{
		SG_SDEBUG("leaving TParameter::copy(): name \"%s\" is different from"
				" target parameter's "
				"name \"%s\"\n", m_name, target->m_name);
		return false;
	}

	SG_SDEBUG("Comparing datatypes without length\n");
	if (!(m_datatype.equals_without_length(target->m_datatype)))
	{
		SG_SDEBUG("leaving TParameter::copy(): type of \"%s\" is different "
				"from target parameter's \"%s\" type\n", m_name, target->m_name);
		return false;
	}

	switch (m_datatype.m_ctype)
	{
		case CT_SCALAR:
		{
			SG_SDEBUG("CT_SCALAR\n");
			if (!TParameter::copy_stype(m_datatype.m_stype,
					m_datatype.m_ptype, m_parameter,
					target->m_parameter))
			{
				SG_SDEBUG("leaving TParameter::copy(): scalar data copy error\n");
				return false;
			}
			break;
		}
		case CT_VECTOR: case CT_SGVECTOR:
		{
			SG_SDEBUG("CT_VECTOR or CT_SGVECTOR\n");

			/* if sizes are different or memory is not allocated, do that */
			if (!m_datatype.equals(target->m_datatype))
			{
				SG_SDEBUG("changing size of target vector and freeing memory\n");
				/* first case: different sizes, free target memory */
				SG_FREE(*(char**)target->m_parameter);
				*(char**)target->m_parameter=NULL;

			}

			/* check whether target m_parameter data contains NULL, if yes
			 * create if the length is non-zero */
			if (*(char**)target->m_parameter==NULL && *m_datatype.m_length_y>0)
			{
				size_t num_bytes=*m_datatype.m_length_y * m_datatype.sizeof_stype();
				SG_SDEBUG("allocating %d bytes memory for target vector\n", num_bytes);

				*(char**)target->m_parameter=SG_MALLOC(char, num_bytes);
				/* check whether ptype is SGOBJECT, if yes we need to initialize
				   the memory with NULL for the way copy_ptype handles it */
				if (m_datatype.m_ptype==PT_SGOBJECT)
					memset(*(void**)target->m_parameter, 0, num_bytes);

				/* use length of source */
				*target->m_datatype.m_length_y=*m_datatype.m_length_y;
			}

			/* now start actual copying, assume that sizes are equal and memory
			 * is there */
			ASSERT(m_datatype.equals(target->m_datatype));

			/* x is number of processed bytes */
			index_t x=0;
			SG_SDEBUG("length_y: %d\n", *m_datatype.m_length_y)
			for (index_t i=0; i<*m_datatype.m_length_y; ++i)
			{
				SG_SDEBUG("copying element %d which is %d byes from start\n",
						i, x);

				void* pointer_a=&((*(char**)m_parameter)[x]);
				void* pointer_b=&((*(char**)target->m_parameter)[x]);

				if (!TParameter::copy_stype(m_datatype.m_stype,
						m_datatype.m_ptype, pointer_a, pointer_b))
				{
					SG_SDEBUG("leaving TParameter::copy(): vector element "
							"copy error\n");
					return false;
				}

				x=x+(m_datatype.sizeof_ptype());
			}

			break;
		}
		case CT_MATRIX: case CT_SGMATRIX:
		{
			SG_SDEBUG("CT_MATRIX or CT_SGMATRIX\n");

			/* if sizes are different or memory is not allocated, do that */
			if (!m_datatype.equals(target->m_datatype))
			{
				SG_SDEBUG("changing size of target vector and freeing memory\n");
				/* first case: different sizes, free target memory */
				SG_FREE(*(char**)target->m_parameter);
				*(char**)target->m_parameter=NULL;
			}

			/* check whether target m_parameter data contains NULL, if yes, create */
			if (*(char**)target->m_parameter==NULL)
			{
				SG_SDEBUG("allocating memory for target vector\n");
				size_t num_bytes=0;
				/* for ST_SPARSE allocate only for a vector of m_length_y */
				if (m_datatype.m_stype==ST_SPARSE)
					num_bytes=*m_datatype.m_length_y * m_datatype.sizeof_stype();
				else
					num_bytes=*m_datatype.m_length_y *
						(*m_datatype.m_length_x) * m_datatype.sizeof_stype();

				*(char**)target->m_parameter=SG_MALLOC(char, num_bytes);

				/* check whether ptype is SGOBJECT, if yes we need to initialize
				   the memory with NULL for the way copy_ptype handles it */
				if (m_datatype.m_ptype==PT_SGOBJECT)
					memset(*(void**)target->m_parameter, 0, num_bytes);

				/* use length of source */
				*target->m_datatype.m_length_y=*m_datatype.m_length_y;
				*target->m_datatype.m_length_x=*m_datatype.m_length_x;

				SG_SDEBUG("%d bytes are allocated\n", num_bytes);
			}

			/* now start actual copying, assume that sizes are equal and memory
			 * is there */
			ASSERT(m_datatype.equals(target->m_datatype));

			/* x is number of processed bytes */
			index_t x=0;
			SG_SDEBUG("length_y: %d\n", *m_datatype.m_length_y)
			SG_SDEBUG("length_x: %d\n", *m_datatype.m_length_x)
			int64_t length=0;
			/* for ST_SPARSE allocate iterate over a vector of m_length_y */
			if (m_datatype.m_stype==ST_SPARSE)
				length=(*m_datatype.m_length_y);
			else
				length=(*m_datatype.m_length_y) * (*m_datatype.m_length_x);
			for (index_t i=0; i<length; ++i)
			{
				SG_SDEBUG("copying element %d which is %d byes from start\n",
						i, x);

				void* pointer_a=&((*(char**)m_parameter)[x]);
				void* pointer_b=&((*(char**)target->m_parameter)[x]);

				if (!TParameter::copy_stype(m_datatype.m_stype,
						m_datatype.m_ptype, pointer_a, pointer_b))
				{
					SG_SDEBUG("leaving TParameter::copy(): vector element "
							"differs\n");
					return false;
				}

				/* For ST_SPARSE, the iteration is on the pointer of SGSparseVectors */
				if (m_datatype.m_stype==ST_SPARSE)
					x=x+(m_datatype.sizeof_stype());
				else
					x=x+(m_datatype.sizeof_ptype());
			}

			break;
		}
		case CT_NDARRAY:
		{
			SG_SDEBUG("CT_NDARRAY\n");
			SG_SERROR("TParameter::copy(): Not yet implemented for "
					"CT_NDARRAY!\n");
			break;
		}
		case CT_UNDEFINED: default:
			SG_SERROR("Implementation error: undefined container type\n");
			break;
	}

	SG_SDEBUG("leaving TParameter::copy(): Copy successful\n");
	return true;
}
