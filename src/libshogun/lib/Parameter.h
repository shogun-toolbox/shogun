/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#include "lib/common.h"
#include "lib/io.h"
#include "lib/DataType.h"
#include "lib/SerialFile.h"
#include "base/DynArray.h"
#include "features/StringFeatures.h"

namespace shogun
{
struct TParameter
{
	explicit TParameter(const TSGDataType* datatype, void* parameter,
						const char* name, const char* description);
	~TParameter(void);

	void print(CIO* io, const char* prefix);
	bool save(CSerialFile* file, const char* prefix="");
	bool load(CSerialFile* file, const char* prefix="");

	TSGDataType m_datatype;
	void* m_parameter;
	char* m_name;
	char* m_description;

private:
	bool is_sgobject(void);
	char* new_prefix(const char* s1, const char* s2);
};

/* Must not be an CSGObject to prevent a recursive call of
 * constructors.
 */
class CParameter
{
	CIO* io;

protected:
	DynArray<TParameter*> m_params;

	virtual void add_type(const TSGDataType* type, void* param,
						  const char* name,
						  const char* description);

public:
	explicit CParameter(CIO* io_);
	virtual ~CParameter(void);

	virtual void print(const char* prefix="");
	virtual bool save(CSerialFile* file, const char* prefix="");
	virtual bool load(CSerialFile* file, const char* prefix="");

	inline virtual int32_t get_num_parameters(void)
	{
		return m_params.get_num_elements();
	}

	/* ************************************************************ */
	/* Scalar wrappers  */

	inline virtual void add_bool(bool* param, const char* name,
								 const char* description="") {
		TSGDataType type(CT_SCALAR, PT_BOOL);
		add_type(&type, param, name, description);
	}

	inline virtual void add_char(char* param, const char* name,
								 const char* description="") {
		TSGDataType type(CT_SCALAR, PT_CHAR);
		add_type(&type, param, name, description);
	}

	inline virtual void add_int16(int16_t* param, const char* name,
								  const char* description="") {
		TSGDataType type(CT_SCALAR, PT_INT16);
		add_type(&type, param, name, description);
	}

	inline virtual void add_int32(int32_t* param, const char* name,
								  const char* description="") {
		TSGDataType type(CT_SCALAR, PT_INT32);
		add_type(&type, param, name, description);
	}

	inline virtual void add_int64(int64_t* param, const char* name,
								  const char* description="") {
		TSGDataType type(CT_SCALAR, PT_INT64);
		add_type(&type, param, name, description);
	}

	inline virtual void add_float32(float32_t* param, const char* name,
									const char* description="") {
		TSGDataType type(CT_SCALAR, PT_FLOAT32);
		add_type(&type, param, name, description);
	}

	inline virtual void add_float64(float64_t* param, const char* name,
									const char* description="") {
		TSGDataType type(CT_SCALAR, PT_FLOAT64);
		add_type(&type, param, name, description);
	}

	inline virtual void add_floatmax(floatmax_t* param, const char* name,
									const char* description="") {
		TSGDataType type(CT_SCALAR, PT_FLOATMAX);
		add_type(&type, param, name, description);
	}

	inline virtual void add_sgobject(CSGObject** param, const char* name,
									 const char* description="") {
		TSGDataType type(CT_SCALAR, PT_SGOBJECT_PTR);
		add_type(&type, param, name, description);
	}

	/* ************************************************************ */
	/* Vector wrappers  */

	inline virtual void add_vector_bool(
		bool** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_VECTOR, PT_BOOL, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_vector_char(
		char** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_VECTOR, PT_CHAR, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_vector_int16(
		int16_t** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_VECTOR, PT_INT16, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_vector_int32(
		int32_t** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_VECTOR, PT_INT32, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_vector_int64(
		int64_t** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_VECTOR, PT_INT64, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_vector_float32(
		float32_t** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_VECTOR, PT_FLOAT32, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_vector_float64(
		float64_t** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_VECTOR, PT_FLOAT64, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_vector_floatmax(
		floatmax_t** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_VECTOR, PT_FLOATMAX, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_vector_sgobject(
		CSGObject** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_VECTOR, PT_SGOBJECT_PTR, length);
		add_type(&type, param, name, description);
	}

	/* ************************************************************ */
	/* Matrix wrappers  */

	inline virtual void add_matrix_bool(
		bool** param, uint64_t length_y, uint64_t length_x,
		const char* name, const char* description="") {
		TSGDataType type(CT_MATRIX, PT_BOOL, length_y, length_x);
		add_type(&type, param, name, description);
	}

	inline virtual void add_matrix_char(
		char** param, uint64_t length_y, uint64_t length_x,
		const char* name, const char* description="") {
		TSGDataType type(CT_MATRIX, PT_CHAR, length_y, length_x);
		add_type(&type, param, name, description);
	}

	inline virtual void add_matrix_int16(
		int16_t** param, uint64_t length_y, uint64_t length_x,
		const char* name, const char* description="") {
		TSGDataType type(CT_MATRIX, PT_INT16, length_y, length_x);
		add_type(&type, param, name, description);
	}

	inline virtual void add_matrix_int32(
		int32_t** param, uint64_t length_y, uint64_t length_x,
		const char* name, const char* description="") {
		TSGDataType type(CT_MATRIX, PT_INT32, length_y, length_x);
		add_type(&type, param, name, description);
	}

	inline virtual void add_matrix_int64(
		int64_t** param, uint64_t length_y, uint64_t length_x,
		const char* name, const char* description="") {
		TSGDataType type(CT_MATRIX, PT_INT64, length_y, length_x);
		add_type(&type, param, name, description);
	}

	inline virtual void add_matrix_float32(
		float32_t** param, uint64_t length_y, uint64_t length_x,
		const char* name, const char* description="") {
		TSGDataType type(CT_MATRIX, PT_FLOAT32, length_y, length_x);
		add_type(&type, param, name, description);
	}

	inline virtual void add_matrix_float64(
		float64_t** param, uint64_t length_y, uint64_t length_x,
		const char* name, const char* description="") {
		TSGDataType type(CT_MATRIX, PT_FLOAT64, length_y, length_x);
		add_type(&type, param, name, description);
	}

	inline virtual void add_matrix_floatmax(
		floatmax_t** param, uint64_t length_y, uint64_t length_x,
		const char* name, const char* description="") {
		TSGDataType type(CT_MATRIX, PT_FLOATMAX, length_y, length_x);
		add_type(&type, param, name, description);
	}

	inline virtual void add_matrix_sgobject(
		CSGObject** param, uint64_t length_y, uint64_t length_x,
		const char* name, const char* description="") {
		TSGDataType type(CT_MATRIX, PT_SGOBJECT_PTR, length_y,
						 length_x);
		add_type(&type, param, name, description);
	}

	/* ************************************************************ */
	/* String wrappers  */

	inline virtual void add_string_bool(
		T_STRING<bool>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_STRING, PT_BOOL, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_string_char(
		T_STRING<char>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_STRING, PT_CHAR, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_string_int16(
		T_STRING<int16_t>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_STRING, PT_INT16, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_string_int32(
		T_STRING<int32_t>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_STRING, PT_INT32, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_string_int64(
		T_STRING<int64_t>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_STRING, PT_INT64, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_string_float32(
		T_STRING<float32_t>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_STRING, PT_FLOAT32, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_string_float64(
		T_STRING<float64_t>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_STRING, PT_FLOAT64, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_string_floatmax(
		T_STRING<floatmax_t>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_STRING, PT_FLOATMAX, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_string_sgobject(
		T_STRING<CSGObject*>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_STRING, PT_SGOBJECT_PTR, length);
		add_type(&type, param, name, description);
	}

	/* ************************************************************ */
	/* Sparse wrappers  */

	inline virtual void add_sparse_bool(
		TSparse<bool>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_SPARSE, PT_BOOL, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_sparse_char(
		TSparse<char>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_SPARSE, PT_CHAR, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_sparse_int16(
		TSparse<int16_t>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_SPARSE, PT_INT16, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_sparse_int32(
		TSparse<int32_t>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_SPARSE, PT_INT32, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_sparse_int64(
		TSparse<int64_t>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_SPARSE, PT_INT64, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_sparse_float32(
		TSparse<float32_t>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_SPARSE, PT_FLOAT32, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_sparse_float64(
		TSparse<float64_t>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_SPARSE, PT_FLOAT64, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_sparse_floatmax(
		TSparse<floatmax_t>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_SPARSE, PT_FLOATMAX, length);
		add_type(&type, param, name, description);
	}

	inline virtual void add_sparse_sgobject(
		TSparse<CSGObject*>** param, uint64_t length, const char* name,
		const char* description="") {
		TSGDataType type(CT_SPARSE, PT_SGOBJECT_PTR, length);
		add_type(&type, param, name, description);
	}
};
}
#endif //__PARAMETER_H__
