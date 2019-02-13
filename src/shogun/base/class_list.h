/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Giovanni De Toni, Yuyu Zhang
 */

#ifndef __SG_CLASS_LIST_H__
#define __SG_CLASS_LIST_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/config.h>

#include <shogun/lib/DataType.h>
#include <shogun/lib/exception/ShogunException.h>

#include <shogun/io/SGIO.h>

#include <set>
#include <string>

namespace shogun {
	class CSGObject;

	/** new shogun instance
	 * @param sgserializable_name
	 * @param generic
	 */
	SHOGUN_EXPORT CSGObject* create(const char* sgserializable_name, EPrimitiveType generic);

	/** deletes object
	 * @param object pointer to object to be deleted
	 */
	SHOGUN_EXPORT void delete_object(CSGObject* object);

	/** Creates new shogun instance, typed.
	 *
	 * Throws an exception in case there is no such classname or
	 * the requested type and the object's type do not match.
	 *
	 */
	template <class T>
	T* create_object(
	    const char* name,
	    EPrimitiveType pt = PT_NOT_GENERIC) throw(ShogunException)
	{
		auto* object = create(name, pt);
		if (!object)
		{
			SG_SERROR(
			    "Class %s with primitive type %s does not exist.\n", name,
			    ptype_name(pt).c_str());
		}
		T* cast = nullptr;
		try
		{
			cast = object->as<T>();
		}
		catch (const ShogunException& e)
		{
			delete_object(object);
			throw e;
		}

		cast->ref();
		return cast;
	}

	/** Returns all available object names
	 *
	 */
	SHOGUN_EXPORT std::set<std::string> available_objects();
}

#endif /* __SG_CLASS_LIST_H__  */
