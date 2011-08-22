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

CTestClass::CTestClass(int32_t number)
{
	m_number=number;
	m_parameters->add(&m_number, "number", "Test variable");

	m_parameter_map->put(
			new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_INT32, 1),
			new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_FLOAT64, 0));
}

TParameter* CTestClass::migrate(DynArray<TParameter*>* param_base,
		SGParamInfo* target)
{
	SG_PRINT("CTestClass::migrate: ");
	SGParamInfo* info=new SGParamInfo("number", CT_SCALAR, ST_NONE, PT_INT32, 1);

	TParameter* result;

	if (*target==*info)
	{
		/* first find index of needed data.
		 * in this case, element in base with same name */
		SGVector<TParameter*> v(param_base->get_array(),
				param_base->get_num_elements());

		/* type is also needed */
		TSGDataType type(target->m_ctype, target->m_stype,
				target->m_ptype);

		/* dummy for searching, search and save result */
		TParameter* t=new TParameter(&type, NULL, target->m_name, "");
		index_t i=CMath::binary_search(v, t);
		TParameter* to_migrate=param_base->get_element(i);
		delete t;

		/* here: simply cast data because nothing has changed */
		int32_t* data=SG_MALLOC(int32_t, 1);
		*data=(int32_t)*((float64_t*)to_migrate->m_parameter);

		/* result structure */
		result=new TParameter(&type, data, to_migrate->m_name,
				to_migrate->m_description);

		SG_PRINT("old: %d, new: %d\n", *((int32_t*)to_migrate->m_parameter),
				*((int32_t*)result->m_parameter));
	}
	else
		result=CSGObject::migrate(param_base, target);

	delete info;

	return result;
}

CTestClass::~CTestClass()
{
	// TODO Auto-generated destructor stub
}

void CTestClass::print()
{
	SG_PRINT("m_number=%d\n", m_number);
}

void CTestClass::load_serializable_pre() throw (ShogunException)
{
	CSGObject::load_serializable_pre();
}
