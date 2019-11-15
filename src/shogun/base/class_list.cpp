/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <shogun/lib/common.h>
#include <shogun/base/class_list.h>
#include <shogun/base/library.h>
#include <shogun/base/SGObject.h>

#include <mutex>

using namespace shogun;
using namespace shogun::io;

static const PluginFactory& shogun::class_list()
{
	static std::mutex kMu;
	static PluginFactory kClassList;

	std::lock_guard<std::mutex> lock(kMu);
	if (kClassList.size() > 0)
		return kClassList;

	for (const auto& plugin: env()->plugins())
	{
		try
		{
			auto library = load_library(plugin);
			try
			{
				for (const auto& class_name: library.manifest().class_list())
				{
					if (kClassList.find(class_name) != kClassList.end())
					{
						io::info("Not registering '{}' class of '{}' as it has been already registered!",
							class_name, plugin);
						continue;
					}

					SG_TRACE("adding '{}' class of '{}' to class list", class_name, plugin);
					kClassList.emplace(class_name, [plugin, class_name]() {
						// TODO: add library handle caching
						auto lib = load_library(plugin);
						return lib.manifest();
						//.class_by_name<SGObject>(class_name);
						//return metaclass.instance();
					});
				}
			}
			catch (std::invalid_argument& e)
			{
				io::warn("Cannot use '{}' as a plugin: {}", plugin, e.what());
			}
			SG_TRACE("Unloading '{}' plugin", plugin);
			unload_library(std::move(library));
		}
		catch (std::invalid_argument& e)
		{
			io::warn("Cannot use {} as a plugin: {}", plugin, e.what());
		}
	}
	return kClassList;
}

std::shared_ptr<SGObject> shogun::create(const std::string& classname, EPrimitiveType generic)
{
	auto clazzes = class_list();
	auto sgo_name = classname + "_sgo";
	auto entry = clazzes.find(sgo_name);
	if (entry != clazzes.end())
	{
		return entry->second().class_by_name<SGObject>(sgo_name).instance();
	}
	return nullptr;
}

std::set<std::string> shogun::available_objects()
{
	std::set<std::string> result;
	for (const auto& each : class_list())
	{
		result.insert(each.first);
	}
	return result;
}
