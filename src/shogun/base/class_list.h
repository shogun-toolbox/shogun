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
	class SGObject;

	/** new shogun instance
	 * @param sgserializable_name
	 * @param generic
	 */
	std::shared_ptr<SGObject> create(const char* sgserializable_name, EPrimitiveType generic);

	/** Creates new shogun instance, typed.
	 *
	 * Throws an exception in case there is no such classname or
	 * the requested type and the object's type do not match.
	 *
	 */
	template <class T>
	std::shared_ptr<T> create_object(
	    const char* name,
	    EPrimitiveType pt = PT_NOT_GENERIC) noexcept(false)
	{
		auto object = create(name, pt);
		if (!object)
		{
			error(
			    "Class {} with primitive type {} does not exist.", name,
			    ptype_name(pt).c_str());
		}
		auto cast = std::dynamic_pointer_cast<T>(object);
		if (cast == nullptr)
		{
			error("could not cst");
		}
		return cast;
	}

	/** Returns all available object names
	 *
	 */
	std::set<std::string> available_objects();
}

#endif /* __SG_CLASS_LIST_H__  */
