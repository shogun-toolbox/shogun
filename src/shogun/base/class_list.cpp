/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Copyright (c) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Written (w) 2009 Soeren Sonnenburg
 * Written (w) 2016 - 2017 Heiko Strathmann
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/lib/common.h>
#include <shogun/base/class_list.h>
#include <shogun/base/library.h>
#include <shogun/base/SGObject.h>

#include <mutex>

using namespace shogun;
using namespace shogun::io;

using CreateFunction = std::function<std::shared_ptr<SGObject>(EPrimitiveType generic)>;
using ClassList = std::unordered_map<std::string, CreateFunction>;

static const ClassList create_class_list() {
	std::mutex mu;
	ClassList class_list;

	std::lock_guard<std::mutex> lock(mu);
	for (const auto& plugin: env()->plugins())
	{
		try
		{
			auto library = load_library(plugin);
			try
			{
				for (const auto& class_name: library.manifest().class_list())
				{
					if (class_list.find(class_name) != class_list.end())
					{
						io::warn("Not registering {} class of {} as it has been already registered!",
							class_name, plugin);
						continue;
					}

					SG_TRACE("adding {} class of {} to class list", class_name, plugin);
					class_list.emplace(class_name, [plugin, class_name](EPrimitiveType generic) {
						// TODO: add library handle caching
						auto lib = load_library(plugin);
						auto metaclass = lib.manifest().class_by_name<SGObject>(class_name);
						return metaclass.instance();
					});
				}
			}
			catch (std::invalid_argument& e)
			{
				io::warn("Cannot use {} as a plugin: {}", plugin, e.what());
			}
			SG_TRACE("Unloading {} plugin", plugin);
			unload_library(std::move(library));
		}
		catch (std::invalid_argument& e)
		{
			io::warn("Cannot use {} as a plugin: {}", plugin, e.what());
		}
	}
	return class_list;
}

ClassList class_list()
{
	static const ClassList kClassList = create_class_list();
	return kClassList;
}

std::shared_ptr<SGObject> shogun::create(const std::string& classname, EPrimitiveType generic)
{
	auto clazzes = class_list();
	auto entry = clazzes.find(classname);
	if (entry != clazzes.end())
	{
		return entry->second(generic);
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
