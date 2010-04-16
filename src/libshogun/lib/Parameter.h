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

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/lib/io.h>
#include <shogun/lib/DataType.h>

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
	const char* name;
	const char* description;
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
		delete m_parameters;
	}

	void add(float64_t* parameter, const char* name, CRange* range=NULL, const char* description=NULL)
	{
		TParameter* par=new TParameter[1];
		par->parameter=parameter;
		par->datatype=DT_SCALAR_REAL;

		if (name)
			par->name=strdup(name);
		else
			par->name=NULL;

		if (description)
			par->description=strdup(description);
		else
			par->description=NULL;

		if (range)
			par->range=range;
		else
			par->range=NULL;

		m_parameters->append_element(par);
	}

	void list_parameters()
	{
	}

	int32_t get_num_parameters()
	{
		return m_parameters->get_num_elements();
	}

	/** @return object name */
	inline virtual const char* get_name() const { return "Parameter"; }

protected:
	CDynamicArray<TParameter*>* m_parameters;
};
}
#endif //__PARAMETER_H__
