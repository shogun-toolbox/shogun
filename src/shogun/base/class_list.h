/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Giovanni De Toni, Yuyu Zhang
 */

#ifndef __SG_CLASS_LIST_H__
#define __SG_CLASS_LIST_H__

#include <shogun/base/SGObject.h>
#include <shogun/base/manifest.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/config.h>
#include <shogun/lib/DataType.h>


#include <set>
#include <string>

namespace shogun {
	class SGObject;

	using CreateFunction = std::function<Manifest()>;
	using PluginFactory = std::unordered_map<std::string, CreateFunction>;

	static inline const PluginFactory& class_list();

	/** new shogun instance
	 * @param sgserializable_name
	 * @param generic
	 */
	SHOGUN_EXPORT std::shared_ptr<SGObject> create(const std::string& sgserializable_name, EPrimitiveType generic);

	/** Creates new shogun instance, typed.
	 *
	 * Throws an exception in case there is no such classname or
	 * the requested type and the object's type do not match.
	 *
	 */
	template <class T>
	std::shared_ptr<T> create_object(
	    const std::string& name,
	    EPrimitiveType pt = PT_NOT_GENERIC) noexcept(false)
	{
		auto clazzes = class_list();
		auto entry = clazzes.find(name);
		if (entry != clazzes.end())
		{
			auto class_factory = entry->second().template class_by_name<T>(name);
			return class_factory.instance();
		}
		else
		{
			error(
			    "Class {} with primitive type {} does not exist.", name,
			    ptype_name(pt).c_str());
		}
		return nullptr;
	}

	/** Returns all available object names
	 *
	 */
	SHOGUN_EXPORT std::set<std::string> available_objects();
}

#endif /* __SG_CLASS_LIST_H__  */
