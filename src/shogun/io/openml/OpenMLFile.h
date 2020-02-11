/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_OPENMLREADER_H
#define SHOGUN_OPENMLREADER_H

#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <numeric>

namespace shogun
{
	/**
	 * Reads OpenML streams which can be downloaded with this function.
	 */
	class OpenMLFile
	{

	public:
		explicit OpenMLFile(const std::string& api_key) : m_api_key(api_key)
		{
		}

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
			std::string request_path;
			auto find_format = m_format_options.find(format);
			if (find_format == m_format_options.end())
			{
				SG_SERROR(
				    "The provided format \"%s\" is not available\n",
				    format.c_str())
			}

			if (format == "split")
			{
				REQUIRE(
				    request == "get_split",
				    "Split server can only handle \"get_split\" request.\n")
				request_path = get_split;
			}
			else
			{
				auto find_request = m_request_options.find(request);
				if (find_request == m_request_options.end())
				{
					SG_SERROR(
					    "Could not find a way to solve the request \"%s\"\n",
					    request.c_str())
				}
				request_path = find_request->second;
			}

			std::string request_format = find_format->second;

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
					    return s0 += "/" + encode_string(s1);
				    });
				request_path += args_string;
			}

			std::string url = request_format + request_path + "?" + m_api_key;

			return get(url);
		}

		std::string get(const std::string& url)
		{
#ifdef HAVE_CURL
			// clear the buffer before request
			m_curl_response_buffer.clear();

			openml_curl_request_helper(url);
			return m_curl_response_buffer;
#else
			SG_SERROR(
			    "Please compile shogun with libcurl to query the OpenML server!\n")
#endif // HAVE_CURL
		}

	private:

		static std::string encode_string(const std::string& s);

		/** the raw buffer as a C++ string */
		std::string m_curl_response_buffer;

		/**
		 * Initialises CURL session and gets the data.
		 * This function also handles the response code from the server.
		 *
		 * @param url the url to query
		 */
		void openml_curl_request_helper(const std::string& url);

		/** the user API key, not required for all requests */
		std::string m_api_key;

		/** the server path to get a response in XML format*/
		static const char* xml_server;
		/** the server path to get a response in JSON format*/
		static const char* json_server;
		/** the server path to download datasets */
		static const char* download_server;
		/** the server path to get a split in ARFF format */
		static const char* splits_server;

		/** the server response format options: XML or JSON */
		static const std::unordered_map<std::string, std::string>
		    m_format_options;
		/** all the supported server options */
		static const std::unordered_map<std::string, std::string>
		    m_request_options;

		/* DATA API */
		static const char* dataset_description;
		static const char* list_data_qualities;
		static const char* data_features;
		static const char* data_qualities;
		static const char* list_dataset_qualities;
		static const char* list_dataset_filter;

		/* FLOW API */
		static const char* flow_file;
		static const char* flow_exists;

		/* TASK API */
		static const char* task_file;

		/* SPLIT API */
		static const char* get_split;
	};

	/**
	 * Writes OpenML streams to the OpenML server.
	 */
	class OpenMLWritter
	{
	public:
		OpenMLWritter(const std::string& api_key) : m_api_key(api_key){};

		template <typename... Args>
		bool post(const std::string& request, const std::string& format, const std::string& message, Args... args);

	private:
		/** the user API key, likely to be needed to write to OpenML */
		std::string m_api_key;
	};
} // namespace shogun

#endif // SHOGUN_OPENMLREADER_H
