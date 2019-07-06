/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Soumyajit De, Jacob Walker,
 *          Thoralf Klein, Sergey Lisitsyn, Bjoern Esser, Viktor Gal,
 *          Weijie Lin, Yori Zwols, Leon Kuchenbecker
 */

#include <string.h>
#include <cctype>

#include <shogun/base/Parameter.h>
#include <shogun/base/class_list.h>
#include <shogun/lib/Hash.h>
#include <shogun/lib/memory.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseMatrix.h>

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
Parameter::add(SGObject** param,
			   const char* name, const char* description) {
	TSGDataType type(CT_SCALAR, ST_NONE, PT_SGOBJECT);
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
Parameter::add_vector(SGObject*** param, index_t* length,
					   const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_SGOBJECT,
					 length);
	add_type(&type, param, name, description);
}


void
Parameter::add_vector(SGVector<bool>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_BOOL, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGVector<char>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_CHAR, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGVector<int8_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_INT8, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGVector<uint8_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_UINT8, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGVector<int16_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_INT16, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGVector<uint16_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_UINT16, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGVector<int32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_INT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGVector<uint32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_UINT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGVector<int64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_INT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGVector<uint64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_UINT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGVector<float32_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_FLOAT32, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGVector<float64_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_FLOAT64, length);
	add_type(&type, param, name, description);
}

void
Parameter::add_vector(SGVector<floatmax_t>** param, index_t* length,
					  const char* name, const char* description) {
	TSGDataType type(CT_VECTOR, ST_NONE, PT_FLOATMAX, length);
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

void Parameter::add(SGVector<SGObject*>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGVECTOR, ST_NONE, PT_SGOBJECT, &param->vlen);
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
	SGObject*** param, index_t* length_y, index_t* length_x,
	const char* name, const char* description) {
	TSGDataType type(CT_MATRIX, ST_NONE, PT_SGOBJECT,
					 length_y, length_x);
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

void Parameter::add(SGMatrix<SGObject*>* param, const char* name,
		const char* description)
{
	TSGDataType type(CT_SGMATRIX, ST_NONE, PT_SGOBJECT, &param->num_rows,
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

void Parameter::add(SGSparseMatrix<SGObject*>* param,
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

bool
TParameter::new_sgserial(SGObject** param,
						 EPrimitiveType generic,
						 const char* sgserializable_name,
						 const char* prefix)
{
	//FIXME
/*	if (*param != NULL)
		SG_UNREF(*param);

	*param = create(sgserializable_name, generic);

	if (*param == NULL) {
		string_t buf = {'\0'};

		if (generic != PT_NOT_GENERIC) {
			buf[0] = '<';
			TSGDataType::ptype_to_string(buf+1, generic,
										 STRING_LEN - 3);
			strcat(buf, ">");
		}

		io::warn("TParameter::new_sgserial(): "
				   "Class `C{}{}' was not listed during compiling Shogun"
				   " :( ...  Can not construct it for `{}{}'!",
				   sgserializable_name, buf, prefix, m_name);

		return false;
	}

	SG_REF(*param);
	*/
	return true;
}

void TParameter::get_incremental_hash(
		uint32_t& hash, uint32_t& carry, uint32_t& total_length)
{

	switch (m_datatype.m_ctype)
	{
	case CT_NDARRAY:
		not_implemented(SOURCE_LOCATION);
		break;
	case CT_SCALAR:
	{
	    uint8_t* data = ((uint8_t*) m_parameter);
		uint32_t size = m_datatype.sizeof_stype();
		total_length += size;
		Hash::IncrementalMurmurHash3(
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
			io::warn("Inconsistency between data structure and "
					"len_y during hashing `{}'!  Continuing with "
					"len_y=0.",
					m_name);
			len_real_y = 0;
		}

		switch (m_datatype.m_ctype)
		{
		case CT_NDARRAY:
			not_implemented(SOURCE_LOCATION);
			break;
		case CT_VECTOR: case CT_SGVECTOR:
			len_real_x = 1;
			break;
		case CT_MATRIX: case CT_SGMATRIX:
			len_real_x = *m_datatype.m_length_x;

			if (*(void**) m_parameter == NULL && len_real_x != 0)
			{
				io::warn("Inconsistency between data structure and "
						"len_x during hashing {}'!  Continuing "
						"with len_x=0.",
						m_name);
				len_real_x = 0;
			}

			if (len_real_x *len_real_y == 0)
				len_real_x = len_real_y = 0;

			break;

		case CT_SCALAR: break;
		case CT_UNDEFINED: default:
			error("Implementation error: undefined container type");
			break;
		}
		uint32_t size = (len_real_x*len_real_y)*m_datatype.sizeof_stype();

		total_length += size;

	        uint8_t* data = (*(uint8_t**) m_parameter);

		Hash::IncrementalMurmurHash3(
				&hash, &carry, data, size);
		break;
	}
	case CT_UNDEFINED: default:
		error("Implementation error: undefined container type");
		break;
	}
}

bool
TParameter::is_valid()
{
	return m_datatype.get_num_elements() > 0;
}

/*
  Initializing m_params(1) with small preallocation-size, because Parameter
  will be constructed several times for EACH SGObject instance.
 */
Parameter::Parameter()
{
	m_params.reserve(1);
}

Parameter::~Parameter()
{
	for (int32_t i=0; i<get_num_parameters(); i++)
		delete m_params[i];
}

void
Parameter::add_type(const TSGDataType* type, void* param,
					 const char* name, const char* description)
{
	if (name == NULL || *name == '\0')
		error("FATAL: Parameter::add_type(): `name' is empty!");

	for (size_t i=0; i<strlen(name); ++i)
	{
		if (!std::isalnum(name[i]) && name[i]!='_' && name[i]!='.')
		{
			error("Character {} of parameter with name \"{}\" is illegal "
					"(only alnum or underscore is allowed)",
					i, name);
		}
	}

	for (int32_t i=0; i<get_num_parameters(); i++)
		if (strcmp(m_params[i]->m_name, name) == 0)
			error("FATAL: Parameter::add_type(): "
					 "Double parameter `{}'!", name);

	m_params.push_back(
		new TParameter(type, param, name, description)
		);
}

void Parameter::set_from_parameters(Parameter* params)
{
	/* iterate over parameters in the given list */
	for (index_t i=0; i<params->get_num_parameters(); ++i)
	{
		TParameter* current=params->get_parameter(i);
		TSGDataType current_type=current->m_datatype;

		ASSERT(m_params.size())

		/* search for own parameter with same name and check types if found */
		TParameter* own=NULL;
		for (index_t j=0; j<m_params.size(); ++j)
		{
			own=m_params[j];
			if (!strcmp(own->m_name, current->m_name))
			{
				if (own->m_datatype==current_type)
				{
					own=m_params[j];
					break;
				}
				else
				{
					index_t l=200;
					char* given_type=SG_MALLOC(char, l);
					char* own_type=SG_MALLOC(char, l);
					current->m_datatype.to_string(given_type, l);
					own->m_datatype.to_string(own_type, l);
					error("given parameter \"{}\" has a different type ({})"
							" than existing one ({})", current->m_name,
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
			error("parameter with name {} does not exist",
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
					SGObject** to_unref=(SGObject**) own->m_parameter;
					SGObject** to_ref=(SGObject**) current->m_parameter;

					if ((*to_ref)!=(*to_unref))
					{
						//FIXME
						//SG_REF((*to_ref));
						//SG_UNREF((*to_unref));
					}

				}
				else
				{
					/* unref all SGObjects and reference the new ones */
					SGObject*** to_unref=(SGObject***) own->m_parameter;
					SGObject*** to_ref=(SGObject***) current->m_parameter;

					for (index_t j=0; j<own->m_datatype.get_num_elements(); ++j)
					{
						if ((*to_ref)[j]!=(*to_unref)[j])
						{
							//FIXME
							//SG_REF(((*to_ref)[j]));
							//SG_UNREF(((*to_unref)[j]));
						}
					}
				}
			}
			else
				error("primitive type PT_SGOBJECT occurred with structure "
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

			/* in case of SGObject, pointers are not equal if SGObjects are
			 * equal, so check. For other values, the pointers are equal and
			 * the not-copying is handled below before the memcpy call */
			if (own->m_datatype.m_ptype==PT_SGOBJECT)
			{
				if (*((SGObject**)dest) == *((SGObject**)source))
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
				dest=*((SGObject**) own->m_parameter);
				source=*((SGObject**) current->m_parameter);
				break;
			default:
				not_implemented(SOURCE_LOCATION);
				break;
			}
		}

		/* copy parameter data, size in memory is equal because of same type */
		if (dest!=source)
			sg_memcpy(dest, source, own->m_datatype.get_size());
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
	for (index_t i=0; i<m_params.size(); ++i)
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

