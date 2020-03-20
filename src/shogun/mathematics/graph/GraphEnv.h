/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben, Viktor Gal
 */

#ifndef SHOGUN_GRAPH_ENV_H_
#define SHOGUN_GRAPH_ENV_H_

#include <shogun/mathematics/graph/shogun-engine_export.h>

#include <mutex>
#include <shared_mutex>
#include <set>
#include <unordered_map>
#include <functional>


namespace shogun
{
	template<typename T>
	class MetaClass;

	namespace graph
	{
		enum class SHOGUN_ENGINE_EXPORT GRAPH_BACKEND
		{
			SHOGUN = 0,
			NGRAPH = 1,
			// TODO: XLA = 2,
			// TODO: TVM = 3
		};

		static const std::unordered_map<GRAPH_BACKEND, std::string_view>
		    kGraphNames = {{GRAPH_BACKEND::SHOGUN, "SHOGUN"},
		                   {GRAPH_BACKEND::NGRAPH, "NGRAPH"}};
		                   // TODO: {GRAPH_BACKEND::XLA, "XLA"},
		                   // TODO: {GRAPH_BACKEND::TVM, "TVM"}};

		class GraphExecutor;

		using CreateExecutor = std::function<MetaClass<GraphExecutor>()>;
		using ExecutorFactory =
		    std::unordered_map<GRAPH_BACKEND, CreateExecutor>;

		static constexpr std::string_view kShogunExecutorName =
		    R"###(.+shogun-[a-z]+-executor\..+)###";

		/** GraphEnv keeps track of all the global settings
		 * of the graph backend, such as which backend executor 
		 * to use. It uses the singleton pattern, so it has static
		 * initialiation and is accessed via GraphEnv::instance().
		 */
		class SHOGUN_ENGINE_EXPORT GraphEnv
		{
			/** The default constructor is private, use 
			 * GraphEnv::instance() instead.
			 * For default values see class member declarations.
			 */
			GraphEnv() = default;

		public:

			// Gives Graph class access to mutex
			friend class Graph;

			~GraphEnv() = default;

			/** Returns the GraphEnv instance from the 
			 * the static initialisation. 
			 */
			static GraphEnv* instance();

			/** Returns the current backend.
			 * Note that this function is locking and thread safe.
			 */
			GRAPH_BACKEND get_backend() const;

			/** Sets the current backend.
			 * Note that this setter is locking and thread safe. 
			 * In the even of creating multiple graphs with 
			 * various bakends this will ensure Graph correctness.
			 * Throws an error if backend not available at 
			 * runtime.
			 * 
			 * @param backend the executor to swicth to
			 */
			void set_backend(const GRAPH_BACKEND backend);

			/** Returns all available object names
			 */
			std::set<GRAPH_BACKEND> available_backends() const;

			/** Returns all available backends and respective
			 * GraphExecutor.
			 */
			const ExecutorFactory& backend_list() const;

		protected:
			/** The shared_mutex to ensure thread safety */
			mutable std::shared_mutex m_env_mutex;

		private:
			/** The backend executor, by default SHOGUN */
			GRAPH_BACKEND m_backend{GRAPH_BACKEND::SHOGUN};
		};
	}
}

#endif