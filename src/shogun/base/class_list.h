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

namespace shogun {
	class CSGObject;

	/** new shogun serializable
	 * @param sgserializable_name
	 * @param generic
	 */
	CSGObject* new_sgserializable(const char* sgserializable_name,
										EPrimitiveType generic);

    /** Creates an instance of a class identified by input name
     * that is wrapped with a shared pointer like structure @ref Some
     *
     * @param name name of class
     * @param type type for template class
     * @return Some object that holds created instance of @ref T
     */
    template<class T>
    T* create(const char* name,
        EPrimitiveType type=PT_NOT_GENERIC)
    {
        return dynamic_cast<T*>(new_sgserializable(name, type));
    }
}

#endif /* __SG_CLASS_LIST_H__  */
