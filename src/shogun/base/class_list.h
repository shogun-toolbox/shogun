/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Giovanni De Toni, Yuyu Zhang
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

	/** deletes object
	 * @param object pointer to object to be deleted
	 */
	void delete_object(CSGObject* object);

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
			delete_object(object);
			SG_SERROR("Type mismatch");
		}
		cast->ref();
		return cast;
	}
}

#endif /* __SG_CLASS_LIST_H__  */
