/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_OPENMLFLOW_H
#define SHOGUN_OPENMLFLOW_H

#include <shogun/io/SGIO.h>

#include <string>
#include <unordered_map>


namespace shogun
{
	/**
	 * Handles OpenML flows. A flow contains the information
	 * required to instantiate a model.
	 */
	class OpenMLFlow
	{

	public:
		/** alias for component type, map of flows */
		using components_type =
		    std::unordered_map<std::string, std::shared_ptr<OpenMLFlow>>;
		/** alias for parameter type, map of maps with information specific to a
		 * parameter */
		using parameters_type = std::unordered_map<
		    std::string, std::unordered_map<std::string, std::string>>;

		/**
		 * The OpenMLFlow constructor. This constructor is rarely used by the
		 * user and is used by the static class members download_flow and
		 * from_file. The user is expected to use either of the previously
		 * mentioned functions.
		 *
		 * @param name the model name
		 * @param description the model description
		 * @param model the flow class_name field
		 * @param components a map of subflows, i.e. kernels
		 * @param parameters a map of parameter information, i.e. default values
		 * for each parameter name
		 */
		OpenMLFlow(
		    const std::string& name, const std::string& description,
		    const std::string& model, components_type components,
		    parameters_type parameters)
		    : m_name(name), m_description(description), m_class_name(model),
		      m_parameters(std::move(parameters)),
		      m_components(std::move(components))
		{
		}

		/**
		 * Instantiates a OpenMLFlow by downloaded a flow from the OpenML
		 * server.
		 *
		 * @param flow_id the flow ID
		 * @param api_key the user API key (might not be required and can be an
		 * empty string)
		 * @return the OpenMLFlow corresponding to the flow requested
		 * @throws ShogunException when there is a server error or the requested
		 * flow is ill formed.
		 */
		static std::shared_ptr<OpenMLFlow>
		download_flow(const std::string& flow_id, const std::string& api_key);

		/**
		 * Instantiates a OpenMLFlow from a file.
		 * @return the OpenMLFlow corresponding to the flow requested
		 */
		static std::shared_ptr<OpenMLFlow> from_file();

		/**
		 * Publishes a flow to the OpenML server
		 * @param flow the flow to be published
		 */
		static void upload_flow(const std::shared_ptr<OpenMLFlow>& flow);

		/**
		 * Dumps the OpenMLFlow to disk.
		 */
		void dump() const;

		/**
		 * Gets a subflow, i.e. a kernel in a machine
		 * @param name the name of the subflow, not the flow ID
		 * @return the subflow if it exists
		 */
		std::shared_ptr<OpenMLFlow> get_subflow(const std::string& name)
		{
			auto find_flow = m_components.find(name);
			if (find_flow != m_components.end())
				return find_flow->second;
			else
				SG_SERROR(
				    "The provided subflow could not be found in this flow!")
			return nullptr;
		}

#ifndef SWIG
		SG_FORCED_INLINE parameters_type

		get_parameters() const noexcept
		{
			return m_parameters;
		}

		SG_FORCED_INLINE components_type

		get_components() const noexcept
		{
			return m_components;
		}

		SG_FORCED_INLINE std::string

		get_class_name() const noexcept
		{
			return m_class_name;
		}

#endif // SWIG

	private:
		/** name field of the flow */
		std::string m_name;
		/** description field of the flow */
		std::string m_description;
		/** the class_name field of the flow */
		std::string m_class_name;
		/** the parameter field of the flow (optional) */
		parameters_type m_parameters;
		/** the components fields of the flow (optional) */
		components_type m_components;
	};
} // namespace shogun

#endif // SHOGUN_OPENMLFLOW_H
