/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011
 */

#ifndef CONVERTER_H_
#define CONVERTER_H_

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/features/Features.h>

namespace shogun
{

/** @brief
 *
 */
class CConverter : public CSGObject
{
public:
	CConverter() : CSGObject() {};
	virtual ~CConverter() {};

	virtual const char* get_name() const { return "Converter"; }

	virtual CFeatures* apply(CFeatures* features) = 0;
};
}
#endif /* CONVERTER_H_ */

