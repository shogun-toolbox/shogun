#include <shogun/base/SGObject.h>
#include <shogun/base/manifest.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/config.h>

#include <shogun/mathematics/graph/OperatorImplementation.h>

#include <set>

namespace shogun {

	static constexpr std::string_view kShogunLinalgPlugin = "shogun-graph";

	using CreateFunction = std::function<Manifest()>;
	using PluginFactory = std::unordered_map<std::string, CreateFunction>;

	static inline const PluginFactory& operator_list();

	/** new operator implementation instance
	 * @param sgserializable_name
	 * @param generic
	 */
	template <class BackendType>
	std::shared_ptr<OperatorImpl<BackendType>> create(const std::string& sgserializable_name);

	/** Creates new shogun instance, typed.
	 *
	 * Throws an exception in case there is no such classname or
	 * the requested type and the object's type do not match.
	 *
	 */
	template <class BackendType>
	std::shared_ptr<OperatorImpl<BackendType>> create_operator(const std::string& name) noexcept(false)
	{
		using BackendImplementation = OperatorImpl<BackendType>;
		auto clazzes = operator_list();
		auto entry = clazzes.find(name);
		if (entry != clazzes.end())
		{
			auto class_factory = entry->second().template class_by_name<BackendImplementation>(name+std::string(BackendImplementation::kBackendName));
			return class_factory.instance();
		}
		else
		{
			error(
			    "Operator {} with backend {} does not exist", name, BackendImplementation::kBackendName);
		}
		return nullptr;
	}

	/** Returns all available object names
	 *
	 */
	std::set<std::string> available_operators();
}