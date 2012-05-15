/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  TREES_H_
#define  TREES_H_

#include <shogun/base/SGObject.h>

namespace shogun
{

class CFeatureTree : public CSGObject
{
public:
	CFeatureTree() : CSGObject()
	{

	}

	~CFeatureTree()
	{

	}

	virtual const char* get_name() const 
	{
		return "FeatureTree";
	}

};
}
#endif   /* ----- #ifndef TREES_H_  ----- */
