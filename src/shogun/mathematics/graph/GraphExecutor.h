#ifndef GRAPH_EXECUTOR_
#define GRAPH_EXECUTOR_

#include <shogun/base/manifest.h>
#include <shogun/mathematics/graph/Tensor.h>
#include <shogun/mathematics/graph/nodes/Node.h>
#include <shogun/mathematics/graph/runtime/RuntimeNode.h>
#include <shogun/mathematics/graph/shogun-engine_export.h>

#include <memory>
#include <regex>
#include <set>
#include <unordered_map>
#include <vector>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{

		static const std::unordered_map<GRAPH_BACKEND, std::string_view>
		    kGraphNames = {{GRAPH_BACKEND::SHOGUN, "SHOGUN"},
		                   {GRAPH_BACKEND::NGRAPH, "NGRAPH"},
		                   {GRAPH_BACKEND::XLA, "XLA"},
		                   {GRAPH_BACKEND::TVM, "TVM"}};

		IGNORE_IN_CLASSLIST class SHOGUN_ENGINE_EXPORT GraphExecutor
		{
		public:
			virtual ~GraphExecutor() = default;
			virtual std::vector<std::shared_ptr<Tensor>> execute(
			    const std::vector<std::shared_ptr<Tensor>>&,
			    const std::vector<std::shared_ptr<node::Node>>&) const = 0;
			virtual void
			add_input_operator(const std::shared_ptr<node::Node>& node) = 0;
			virtual void
			add_operator_node(const std::shared_ptr<node::Node>& node) = 0;
			void set_requires_major_conversion(bool requires_major_conversion)
			{
				m_requires_major_conversion = requires_major_conversion;
			}

		protected:
			std::vector<std::shared_ptr<detail::RuntimeNode>>
			    m_cached_input_operators;
			std::vector<std::shared_ptr<detail::RuntimeNode>>
			    m_cached_operators;
			bool m_requires_major_conversion = false;
		};

		static constexpr std::string_view kShogunExecutorName =
		    R"###(.+shogun-[a-z]+-executor\..+)###";

		using CreateExecutor = std::function<MetaClass<GraphExecutor>()>;
		using ExecutorFactory =
		    std::unordered_map<GRAPH_BACKEND, CreateExecutor>;

		SHOGUN_ENGINE_EXPORT const ExecutorFactory& backend_list();

		/** new operator implementation instance
		 * @param backend_name
		 * @param generic
		 */
		SHOGUN_ENGINE_EXPORT std::shared_ptr<GraphExecutor>
		create(GRAPH_BACKEND backend_type);

		/** Returns all available object names
		 *
		 */
		SHOGUN_ENGINE_EXPORT std::set<GRAPH_BACKEND> available_backends();

#define BEGIN_EXECUTOR_MANIFEST(DESCRIPTION)                                   \
	extern "C" ::shogun::Manifest shogunManifest()                             \
	{                                                                          \
		    static ::shogun::Manifest manifest(DESCRIPTION,{

/** Declares class to be exported.
 * Always use this macro between @ref BEGIN_MANIFEST and
 * @ref END_MANIFEST
 */
#define EXPORT_EXECUTOR(CLASSNAME)                                             \
	std::make_pair(                                                            \
	    std::string(kGraphNames.at(CLASSNAME::kBackendType)),                  \
	    ::shogun::make_any(::shogun::MetaClass<GraphExecutor>(                 \
	        ::shogun::make_any<std::shared_ptr<GraphExecutor>>(                \
	            []() -> std::shared_ptr<GraphExecutor> {                       \
		            return std::make_shared<CLASSNAME>();                      \
	            })))),

/** Ends manifest declaration.
 * Always use this macro after @ref BEGIN_MANIFEST
 */
#define END_EXECUTOR_MANIFEST()                                                \
	});                                                                        \
	return manifest;                                                           \
	}
	} // namespace graph
} // namespace shogun

#endif /* GRAPH_EXECUTOR_ */
