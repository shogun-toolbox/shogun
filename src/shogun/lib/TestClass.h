/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __TESTCLASS_H_
#define __TESTCLASS_H_

#include <shogun/base/SGObject.h>

namespace shogun
{

class CTestClass : public CSGObject
{
public:
	CTestClass() {}
	CTestClass(int32_t number);
	virtual ~CTestClass();

	void print();

	inline virtual const char* get_name() const { return "TestClass"; }

protected:
	TParameter* migrate(DynArray<TParameter*>* param_base, SGParamInfo* element);
	void load_serializable_pre() throw (ShogunException);

protected:
	int32_t m_number;

};

}

#endif /* __TESTCLASS_H_ */
