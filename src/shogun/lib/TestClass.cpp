/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/lib/TestClass.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/ParameterMap.h>

using namespace shogun;

CTestClass::CTestClass()
{
	m_number=10;
	m_parameters->add(&m_number, "new", "Test variable");

	m_parameter_map->put(
			new SGParamInfo("new", CT_SCALAR, ST_NONE, PT_FLOAT64, 1),
			new SGParamInfo("old", CT_SCALAR, ST_NONE, PT_INT32, 0));
}

CTestClass::~CTestClass()
{
	// TODO Auto-generated destructor stub
}

void CTestClass::print()
{
	SG_PRINT("m_number=%d\n", m_number);
}
