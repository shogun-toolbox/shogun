/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/io/openml/OpenMLFile.h>

#ifdef HAVE_CURL
#include "OpenMLFile.h"
#include <curl/curl.h>

#endif // HAVE_CURL

using namespace shogun;

/**
 * The writer callback function used to write the packets to a C++ string.
 * @param data the data received in CURL request
 * @param size always 1
 * @param nmemb the size of data
 * @param buffer_in the buffer to write to
 * @return the size of buffer that was written
 */
size_t writer(char* data, size_t size, size_t nmemb, std::string* buffer_in)
{
	// check that the buffer string points to something
	if (buffer_in != nullptr)
	{
		// Append the data to the buffer
		buffer_in->append(data, size * nmemb);

		return size * nmemb;
	}
	return 0;
}

/* OpenML server format */
const char* OpenMLFile::xml_server = "https://www.openml.org/api/v1/xml";
const char* OpenMLFile::json_server = "https://www.openml.org/api/v1/json";
const char* OpenMLFile::download_server = "";
const char* OpenMLFile::splits_server = "https://www.openml.org/api_splits";

/* DATA API */
const char* OpenMLFile::dataset_description = "/data/{}";
const char* OpenMLFile::list_data_qualities = "/data/qualities/list";
const char* OpenMLFile::data_features = "/data/features/{}";
const char* OpenMLFile::data_qualities = "/data/qualities/{}";
const char* OpenMLFile::list_dataset_qualities = "/data/qualities/{}";
const char* OpenMLFile::list_dataset_filter = "/data/list/{}";
/* FLOW API */
const char* OpenMLFile::flow_file = "/flow/{}";
const char* OpenMLFile::flow_exists = "/flow/exists/{}";
/* TASK API */
const char* OpenMLFile::task_file = "/task/{}";
/* SPLIT API */
const char* OpenMLFile::get_split = "";

const std::unordered_map<std::string, std::string>
    OpenMLFile::m_format_options = {{"xml", xml_server},
                                    {"json", json_server},
                                    {"split", splits_server},
                                    {"download", download_server}};
const std::unordered_map<std::string, std::string>
    OpenMLFile::m_request_options = {
        {"dataset_description", dataset_description},
        {"list_data_qualities", list_data_qualities},
        {"data_features", data_features},
        {"data_qualities", data_qualities},
        {"list_dataset_qualities", list_dataset_qualities},
        {"list_dataset_filter", list_dataset_filter},
        {"flow_file", flow_file},
        {"flow_exists", flow_exists},
        {"task_file", task_file}};

void OpenMLFile::openml_curl_request_helper(const std::string& url)
{
#ifdef HAVE_CURL
	auto curl_handle = curl_easy_init();

	if (!curl_handle)
	{
		SG_SERROR("Failed to initialise curl handle.\n")
	}

	curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
	curl_easy_setopt(curl_handle, CURLOPT_HTTPGET, 1);
	curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, writer);
	curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &m_curl_response_buffer);

	CURLcode res = curl_easy_perform(curl_handle);

	if (res != CURLE_OK)
		SG_SERROR("Connection error: %s.\n", curl_easy_strerror(res))

	curl_easy_cleanup(curl_handle);
#endif // HAVE_CURL
}

std::string OpenMLFile::encode_string(const std::string& s)
{
#ifdef HAVE_CURL
	auto curl_handle = curl_easy_init();

	if (!curl_handle)
	{
		SG_SERROR("Failed to initialise curl handle.\n")
	}

	char* encoded_url = curl_easy_escape(curl_handle, s.c_str(), s.size());
	if (!encoded_url)
		SG_SERROR("Failed to encode \"%s\" URL escaped.\n", s.c_str())
	return encoded_url;
#else
	return s;
#endif
}
