/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben, Viktor Gal
 */

#include "GraphEnv.h"
#include "GraphExecutor.h"

#include <shogun/io/SGIO.h>

#include <regex>

using namespace shogun::graph;

GraphEnv* GraphEnv::instance()
{
	static GraphEnv result{};
	return &result;
}

GRAPH_BACKEND GraphEnv::get_backend() const
{
	std::shared_lock lock(m_env_mutex);
	return m_backend;
}

void GraphEnv::set_backend(const GRAPH_BACKEND backend)
{
	if (available_backends().count(backend))
	{
		// mutex is locked here to avoid having graphs with mixed backend ops
		std::unique_lock lock(m_env_mutex);
		m_backend = backend;
	}
	else
		error("{} backend not available.", kGraphNames.at(backend));
}

std::set<GRAPH_BACKEND> GraphEnv::available_backends() const
{
	std::set<GRAPH_BACKEND> result;
	for (const auto& each : backend_list())
	{
		result.insert(each.first);
	}
	return result;
}

const ExecutorFactory& GraphEnv::backend_list() const
{
	static std::mutex kMu;
	static ExecutorFactory kBackendList;

	std::lock_guard<std::mutex> lock(kMu);
	if (!kBackendList.empty())
		return kBackendList;

	const std::regex executor_regex(kShogunExecutorName.data());
	for (const auto& plugin : env()->plugins())
	{
		if (!std::regex_match(plugin, executor_regex))
			continue;
		try
		{
			auto library = load_library(plugin);
			try
			{
				for (const auto& backend_name : library.manifest().class_list())
				{
					GRAPH_BACKEND backend_type;
					for (auto&& kv : kGraphNames)
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
						io::info(
						    "Not registering '{}' class of '{}' as it has been "
						    "already registered!",
						    backend_name, plugin);
						continue;
					}

					SG_TRACE(
					    "adding '{}' class of '{}' to backend list",
					    backend_name, plugin);
					kBackendList.emplace(
					    backend_type, [plugin, backend_name]() {
						    // TODO: add library handle caching
						    auto lib = load_library(plugin);
						    return lib.manifest().class_by_name<GraphExecutor>(
						        backend_name);
					    });
				}
			}
			catch (std::invalid_argument& e)
			{
				io::warn(
				    "Cannot use '{}' as a graph backend: {}", plugin, e.what());
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