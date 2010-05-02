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

class CRange
{
	public:
	CRange()
	{
	}

	~CRange()
	{
	}

protected:
	double lower_limit;
	double default_value;
	double upper_limit;
};

struct TParameter
{
	void* parameter;
	SGDataType datatype;
	char* name;
	char* description;
	/** valid range */
	CRange* range;
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

	void add(float64_t* parameter, const char* name,
			CRange* range=NULL, const char* description=NULL);

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
