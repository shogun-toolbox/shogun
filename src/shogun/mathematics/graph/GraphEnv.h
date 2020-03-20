/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben, Viktor Gal
 */

#ifndef SHOGUN_GRAPH_ENV_H_
#define SHOGUN_GRAPH_ENV_H_

#include <mutex>
#include <set>
#include <unordered_map>
#include <functional>


namespace shogun
{
	template<typename T>
	class MetaClass;

	namespace graph
	{
		enum class GRAPH_BACKEND
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

		class GraphEnv
		{
			GraphEnv()
			{
			}

		public:

			friend class Graph;

			~GraphEnv() = default;

			static GraphEnv* instance()
			{
				static GraphEnv result{};
				return &result;
			}

			GRAPH_BACKEND get_backend() const;

			void set_backend(const GRAPH_BACKEND backend);

			/** Returns all available object names
			 *
			 */
			std::set<GRAPH_BACKEND> available_backends() const;

			const ExecutorFactory& backend_list() const;

		protected:
			std::mutex m_env_mutex;

		private:
			GRAPH_BACKEND m_backend = GRAPH_BACKEND::SHOGUN;
		};
	}
}

#endif