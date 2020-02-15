#include <shogun/mathematics/graph/GraphExecutor.h>

#include <mutex>

using namespace shogun;
using namespace shogun::io;

const ExecutorFactory& shogun::backend_list()
{
	static std::mutex kMu;
	static ExecutorFactory kBackendList;

	std::lock_guard<std::mutex> lock(kMu);
	if (!kBackendList.empty())
		return kBackendList;

	for (const auto& plugin: env()->plugins())
	{
		if (plugin.find(kShogunExecutorName) == std::string::npos)
			continue;
		try
		{
			auto library = load_library(plugin);
			try
			{
				for (const auto& backend_name: library.manifest().class_list())
				{
                    GRAPH_BACKEND backend_type;
                    for (auto&& kv: kGraphNames)
                    {
                        if (kv.second == backend_name)
                        {
                            backend_type = kv.first;
                            break;
                        }
                    }
                    // TODO: what if it's not in the list?

					if (kBackendList.find(backend_type) != kBackendList.end())
					{
						io::info("Not registering '{}' class of '{}' as it has been already registered!",
							backend_name, plugin);
						continue;
					}

					SG_TRACE("adding '{}' class of '{}' to backend list", backend_name, plugin);
					kBackendList.emplace(backend_type, [plugin, backend_name]() {
						// TODO: add library handle caching
						auto lib = load_library(plugin);
						return lib.manifest().class_by_name<GraphExecutor>(backend_name);
					});
				}
			}
			catch (std::invalid_argument& e)
			{
				io::warn("Cannot use '{}' as a graph backend: {}", plugin, e.what());
			}
			SG_TRACE("Unloading '{}' graph backend", plugin);
			unload_library(std::move(library));
		}
		catch (std::invalid_argument& e)
		{
			io::warn("Cannot use {} as a graph backend: {}", plugin, e.what());
		}
	}
	if (kBackendList.empty())
		error("No graph execution backend found, ABORTING CAPTAIN.");
	return kBackendList;
}

std::shared_ptr<GraphExecutor> shogun::create(GRAPH_BACKEND backend)
{
	auto backends = backend_list();
	auto entry = backends.find(backend);
	if (entry != backends.end())
	{
		return entry->second().instance();
	}
	return nullptr;
}

std::set<GRAPH_BACKEND> shogun::available_backends()
{
	std::set<GRAPH_BACKEND> result;
	for (const auto& each : backend_list())
	{
		result.insert(each.first);
	}
	return result;
}
