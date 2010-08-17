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

#include "base/SGObject.h"
#include "lib/common.h"
#include "lib/DynamicArray.h"
#include "lib/io.h"
#include "lib/DataType.h"

namespace shogun
{

struct TParameter
{
	void* parameter;
	SGDataType datatype;
	char* name;
	char* description;

	/** valid range - minimum */
	union
	{
		float32_t min_value_float32;
		float64_t min_value_float64;
	};

	/** valid range - maximum */
	union
	{
		float32_t max_value_float32;
		float64_t max_value_float64;
	};
};

class CParameter: public CSGObject
{
public:
	CParameter()
	{
		m_parameters = new CDynamicArray<TParameter*>();
	}

	/** default destructor */
	virtual ~CParameter()
	{
		free_parameters();
	}

	void add_float(float32_t* parameter, const char* name,
			const char* description=NULL,
			float64_t min_value=-CMath::INFTY, float64_t max_value=CMath::INFTY);

	void add_double(float64_t* parameter, const char* name,
			const char* description=NULL, float64_t min_value=-CMath::INFTY,
			float64_t max_value=CMath::INFTY);

	void add_float_vector(float32_t** parameter, int64_t length,
			const char* name, const char* description=NULL,
			float64_t min_value=-CMath::INFTY, float64_t max_value=CMath::INFTY);

	void add_double_vector(float64_t** parameter, int64_t length,
			const char* name, const char* description=NULL,
			float64_t min_value=-CMath::INFTY, float64_t max_value=CMath::INFTY);

	void add_sgobject(CSGObject* parameter, const char* name, const char* description=NULL);

	void list_parameters();

	inline int32_t get_num_parameters()
	{
		return m_parameters->get_num_elements();
	}

	/** @return object name */
	inline virtual const char* get_name() const { return "Parameter"; }

protected:
	void free_parameters();

protected:
	CDynamicArray<TParameter*>* m_parameters;
};
}
#endif //__PARAMETER_H__
