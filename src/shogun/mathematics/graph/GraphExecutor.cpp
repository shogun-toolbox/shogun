/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <shogun/mathematics/graph/GraphExecutor.h>
#include "GraphEnv.h"

using namespace shogun::graph;
using namespace shogun::io;


std::shared_ptr<GraphExecutor> shogun::graph::create(GRAPH_BACKEND backend)
{
	auto backends = GraphEnv::instance()->backend_list();
	auto entry = backends.find(backend);
	if (entry != backends.end())
	{
		return entry->second().instance();
	}
	return nullptr;
}
