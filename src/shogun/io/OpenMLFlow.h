/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_OPENMLFLOW_H
#define SHOGUN_OPENMLFLOW_H

#include <shogun/lib/config.h>

#ifdef HAVE_CURL

#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>

#include <curl/curl.h>

#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace shogun
{
	class OpenMLReader
	{

	public:
		explicit OpenMLReader(const std::string& api_key);

		/**
		 * Returns a string returned by the server given a request.
		 * Raises an error if the returned code is not 200.
		 * Additional arguments can be passed to the request,
		 * which are then concatenated with a "/" character.
		 *
		 * @tparam Args argument type pack, should all be std::string
		 * @param request the request name, see m_request_options
		 * @param format the format to return the data in, see m_format_options
		 * @param args the additional arguments to be passed to request
		 * @return the returned stream from the server if the return code is 200
		 */
		template <typename... Args>
		std::string
		get(const std::string& request, const std::string& format, Args... args)
		{
			auto find_format = m_format_options.find(format);
			if (find_format == m_format_options.end())
			{
				SG_SERROR(
				    "The provided format \"%s\" is not available\n",
				    format.c_str())
			}
			auto find_request = m_request_options.find(request);
			if (find_request == m_request_options.end())
			{
				SG_SERROR(
				    "Could not find a way to solve the request \"%s\"\n",
				    request.c_str())
			}
			std::string request_format = find_format->second;
			std::string request_path = find_request->second;

			// get additional args and concatenate them with "/"
			if (sizeof...(Args) > 0)
			{
				if (request_path.substr(request_path.size() - 2) == "{}")
				{
					request_path =
					    request_path.substr(0, request_path.size() - 2);
				}
				else
				{
					SG_SERROR(
					    "The provided request \"%s\" cannot handle additional "
					    "args.\n",
					    request.c_str())
				}
				std::vector<std::string> args_vec = {args...};
				std::string args_string = std::accumulate(
				    args_vec.begin() + 1, args_vec.end(), args_vec.front(),
				    [](std::string s0, std::string& s1) {
					    return s0 += "/" + s1;
				    });
				request_path += args_string;
			}

			std::string url = request_format + request_path + "?" + m_api_key;

			openml_curl_request_helper(url);

			return m_curl_response_buffer;
		}

	private:
		std::string m_curl_response_buffer;

		/**
		 * Initialises CURL session and gets the data.
		 * This function also handles the response code from the server.
		 *
		 * @param url the url to query
		 */
		void openml_curl_request_helper(const std::string& url);

		/**
		 * Handles all possible codes
		 *
		 * @param curl_handle curl handle used in the request
		 * @param code the code returned by the query
		 */
		void openml_curl_error_helper(CURL* curl_handle, CURLcode code);

		std::string m_api_key;

		static const char* xml_server;
		static const char* json_server;

		static const std::unordered_map<std::string, std::string>
		    m_format_options;
		static const std::unordered_map<std::string, std::string>
		    m_request_options;

		/* DATA API */
		static const char* dataset_description;
		static const char* list_data_qualities;
		static const char* data_features;
		static const char* list_dataset_qualities;
		static const char* list_dataset_filter;

		/* FLOW API */
		static const char* flow_file;
	};

	class OpenMLWritter
	{
	public:
		OpenMLWritter(const std::string& api_key) : m_api_key(api_key){};

	private:
		std::string m_api_key;
	};

	class OpenMLFlow
	{

	public:
		using components_type =
		    std::unordered_map<std::string, std::shared_ptr<OpenMLFlow>>;
		using parameters_type = std::unordered_map<
		    std::string, std::unordered_map<std::string, std::string>>;

		OpenMLFlow(
		    const std::string& name, const std::string& description,
		    const std::string& model, components_type components,
		    parameters_type parameters)
		    : m_name(name), m_description(description), m_class_name(model),
		      m_parameters(parameters), m_components(components)
		{
		}

		static std::shared_ptr<OpenMLFlow>
		download_flow(const std::string& flow_id, const std::string& api_key);

		static std::shared_ptr<OpenMLFlow> from_file();

		static void upload_flow(const std::shared_ptr<OpenMLFlow>& flow);

		void dump();

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
		SG_FORCED_INLINE parameters_type get_parameters()
		{
			return m_parameters;
		}

		SG_FORCED_INLINE components_type get_components()
		{
			return m_components;
		}

		SG_FORCED_INLINE std::string get_class_name()
		{
			return m_class_name;
		}
#endif // SWIG

	private:
		std::string m_name;
		std::string m_description;
		std::string m_class_name;
		parameters_type m_parameters;
		components_type m_components;
	};

#ifndef SWIG
	template <typename T>
	T char_to_scalar(const char* string_val)
	{
		SG_SERROR(
		    "No registered conversion from string to type \"s\"\n",
		    demangled_type<T>().c_str())
		return 0;
	}

	template <>
	float32_t char_to_scalar(const char* string_val)
	{
		char* end;
		return std::strtof(string_val, &end);
	}

	template <>
	float64_t char_to_scalar(const char* string_val)
	{
		char* end;
		return std::strtod(string_val, &end);
	}

	template <>
	floatmax_t char_to_scalar(const char* string_val)
	{
		char* end;
		return std::strtold(string_val, &end);
	}

	template <>
	bool char_to_scalar(const char* string_val)
	{
		return strcmp(string_val, "true");
	}

#endif // SWIG

	class ShogunOpenML
	{
	public:
		static std::shared_ptr<CSGObject> flow_to_model(
		    std::shared_ptr<OpenMLFlow> flow, bool initialize_with_defaults);

		static std::shared_ptr<OpenMLFlow>
		model_to_flow(const std::shared_ptr<CSGObject>& model);

	private:
		static std::tuple<std::string, std::string>
		get_class_info(const std::string& class_name);
	};
} // namespace shogun
#endif // HAVE_CURL

#endif // SHOGUN_OPENMLFLOW_H
