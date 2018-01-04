/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society
 */

#ifndef __SG_CLASS_LIST_H__
#define __SG_CLASS_LIST_H__

#include <shogun/lib/config.h>

#include <shogun/lib/DataType.h>

#include <shogun/io/SGIO.h>

namespace shogun {
	class CSGObject;

	/** new shogun instance
	 * @param sgserializable_name
	 * @param generic
	 */
	CSGObject* create(const char* sgserializable_name, EPrimitiveType generic);

	/** Creates new shogun instance, typed.
	 *
	 * Throws an exception in case there is no such classname or
	 * the requested type and the object's type do not match.
	 *
	 */
	template <class T>
	T* create_object(const char* name)
	{
		auto* object = create(name, PT_NOT_GENERIC);
		if (!object)
		{
			SG_SERROR("No such class %s", name);
		}
		auto* cast = dynamic_cast<T*>(object);
		if (!cast)
		{
			delete object;
			SG_SERROR("Type mismatch");
		}
		cast->ref();
		return cast;
	}
}

#endif /* __SG_CLASS_LIST_H__  */
