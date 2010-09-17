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
#include "lib/DynamicArray.h"
#include "base/SGObject.h"

namespace shogun
{
struct TParameter
{
	explicit TParameter(const TSGDataType* datatype, void* parameter,
						const char* name, const char* description);
	~TParameter(void);

	TSGDataType m_datatype;
	void* m_parameter;
	char* m_name;
	char* m_description;

private:
	bool is_sgobject(void);
};

class CParameter :CSGObject
{
protected:
	CDynamicArray<TParameter*> m_parameters;

	virtual void add_type(const TSGDataType* type, void* param,
						  const char* name,
						  const char* description=NULL);

public:
	explicit CParameter(void);
	virtual ~CParameter(void);

	/** @return object name */
	inline virtual const char* get_name(void) const
	{
		return "Parameter";
	}

	virtual void list_parameters(void);

	inline virtual int32_t get_num_parameters(void)
	{
		return m_parameters.get_num_elements();
	}

	/* ************************************************************ */
	/* Scalar wrappers  */

	inline virtual void add_bool(bool* param, const char* name,
								 const char* description=NULL) {
		TSGDataType type(CT_SCALAR, PT_BOOL);
		add_type(&type, param, name, description);
	}

	inline virtual void add_char(char* param, const char* name,
								 const char* description=NULL) {
		TSGDataType type(CT_SCALAR, PT_CHAR);
		add_type(&type, param, name, description);
	}

	inline virtual void add_int16(int16_t* param, const char* name,
								  const char* description=NULL) {
		TSGDataType type(CT_SCALAR, PT_INT16);
		add_type(&type, param, name, description);
	}

	inline virtual void add_int32(int32_t* param, const char* name,
								  const char* description=NULL) {
		TSGDataType type(CT_SCALAR, PT_INT32);
		add_type(&type, param, name, description);
	}

	inline virtual void add_int64(int64_t* param, const char* name,
								  const char* description=NULL) {
		TSGDataType type(CT_SCALAR, PT_INT64);
		add_type(&type, param, name, description);
	}

	inline virtual void add_float32(float32_t* param, const char* name,
									const char* description=NULL) {
		TSGDataType type(CT_SCALAR, PT_FLOAT32);
		add_type(&type, param, name, description);
	}

	inline virtual void add_float64(float64_t* param, const char* name,
									const char* description=NULL) {
		TSGDataType type(CT_SCALAR, PT_FLOAT64);
		add_type(&type, param, name, description);
	}

	inline virtual void add_floatmax(floatmax_t* param, const char* name,
									const char* description=NULL) {
		TSGDataType type(CT_SCALAR, PT_FLOATMAX);
		add_type(&type, param, name, description);
	}

	inline virtual void add_sgobject(CSGObject** param, const char* name,
									 const char* description=NULL) {
		TSGDataType type(CT_SCALAR, PT_SGOBJECT_PTR);
		add_type(&type, param, name, description);
	}
};
}
#endif //__PARAMETER_H__
