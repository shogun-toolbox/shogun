#include <shogun/mathematics/graph/operator_list.h>

#include <mutex>

using namespace shogun;
using namespace shogun::io;

#include <iostream>

const PluginFactory& shogun::operator_list()
{
	static std::mutex kMu;
	static PluginFactory kOperatorList;

	std::lock_guard<std::mutex> lock(kMu);
	if (!kOperatorList.empty())
		return kOperatorList;

	for (const auto& plugin: env()->plugins())
	{
		if (plugin.find(kShogunLinalgPlugin) == std::string::npos)
			continue;
		try
		{
			auto library = load_library(plugin);
			try
			{
				for (const auto& operator_name: library.manifest().class_list())
				{
					if (kOperatorList.find(operator_name) != kOperatorList.end())
					{
						io::info("Not registering '{}' class of '{}' as it has been already registered!",
							operator_name, plugin);
						continue;
					}

					SG_TRACE("adding '{}' class of '{}' to class list", operator_name, plugin);
					kOperatorList.emplace(operator_name, [plugin, operator_name]() {
						// TODO: add library handle caching
						auto lib = load_library(plugin);
						return lib.manifest();
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
	if (kOperatorList.empty())
		error("No operators found, ABORTING CAPTAIN.");
	return kOperatorList;
}

template <class BackendType>
std::shared_ptr<OperatorImpl<BackendType>> shogun::create(const std::string& operator_name)
{
	auto operators = operator_list();
	auto sgo_name = operator_name + BackendType::kBackendName;
	auto entry = operators.find(sgo_name);
	if (entry != operators.end())
	{
		return entry->second().template class_by_name<OperatorImpl<BackendType>>(sgo_name);
	}
	return nullptr;
}

std::set<std::string> shogun::available_operators()
{
	std::set<std::string> result;
	for (const auto& each : operator_list())
	{
		result.insert(each.first);
	}
	return result;
}
