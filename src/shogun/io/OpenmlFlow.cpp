/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifdef HAVE_CURL

#include <shogun/io/OpenmlFlow.h>
#include "OpenmlFlow.h"


using namespace shogun;

size_t writer(char *data, size_t size, size_t nmemb, std::string* buffer_in)
{
	// adapted from https://stackoverflow.com/a/5780603
	// Is there anything in the buffer?
	if (buffer_in->empty())
	{
		// Append the data to the buffer
		buffer_in->append(data, size * nmemb);

		return size * nmemb;
	}

	return 0;
}

const char* OpenMLReader::xml_server = "https://www.openml.org/api/v1/xml";
const char* OpenMLReader::json_server = "https://www.openml.org/api/v1/json";
const char* OpenMLReader::dataset_description = "/data/{}";
const char* OpenMLReader::list_data_qualities = "/data/qualities/list";
const char* OpenMLReader::data_features = "/data/features/{}";
const char* OpenMLReader::list_dataset_qualities = "/data/qualities/{}";
const char* OpenMLReader::list_dataset_filter = "/data/list/{}";
const char* OpenMLReader::flow_file = "/flow/{}";

const std::unordered_map<std::string, std::string>
    OpenMLReader::m_format_options = {{"xml", xml_server},
                                      {"json", json_server}};
const std::unordered_map<std::string, std::string>
    OpenMLReader::m_request_options = {

        {"dataset_description", dataset_description},
        {"list_data_qualities", list_data_qualities},
        {"data_features", data_features},
        {"list_dataset_qualities", list_dataset_qualities},
        {"list_dataset_filter", list_dataset_filter},
		{"flow_file", flow_file}};

OpenMLReader::OpenMLReader(const std::string& api_key) : m_api_key(api_key)
{
}

void OpenMLReader::post(const std::string& request, const std::string& data)
{
}

void OpenMLReader::openml_curl_request_helper(const std::string& url)
{
	CURL* curl_handle = nullptr;

	curl_handle = curl_easy_init();

	if (!curl_handle)
	{
		SG_SERROR("Failed to initialise curl handle.")
		return;
	}

	curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
	curl_easy_setopt(curl_handle, CURLOPT_HTTPGET,1);
	curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, writer);
	curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &m_curl_response_buffer);

	CURLcode res = curl_easy_perform(curl_handle);

	openml_curl_error_helper(res);

	curl_easy_cleanup(curl_handle);
}

void OpenMLReader::openml_curl_error_helper(CURLcode code) {

}


void OpenMLFlow::download_flow()
{

	auto reader = OpenMLReader(m_api_key);
	auto return_string = reader.get("flow_file", "json", m_flow_id);
}

void OpenMLFlow::upload_flow(const OpenMLFlow& flow)
{
}

#endif // HAVE_CURL
