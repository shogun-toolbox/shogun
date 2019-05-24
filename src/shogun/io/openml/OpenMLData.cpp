/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/RegressionLabels.h>

#include <shogun/io/openml/OpenMLData.h>
#include <shogun/io/openml/OpenMLFile.h>
#include <shogun/io/openml/utils.h>

using namespace shogun;
using namespace shogun::openml_detail;
using namespace rapidjson;

std::shared_ptr<OpenMLData>
OpenMLData::get_dataset(const std::string& id, const std::string& api_key)
{
	// description
	auto reader = OpenMLFile(api_key);
	auto return_string = reader.get("dataset_description", "json", id);

	auto& dataset_description = check_response<BACKEND_FORMAT::JSON>(
	    return_string, "data_set_description");

	auto name = return_if_possible<std::string>(
	    "name", dataset_description.GetObject());
	auto description = return_if_possible<std::string>(
	    "description", dataset_description.GetObject());
	auto data_format = return_if_possible<std::string>(
	    "data_format", dataset_description.GetObject());
	auto dataset_id =
	    return_if_possible<std::string>("id", dataset_description.GetObject());
	auto version = return_if_possible<std::string>(
	    "version", dataset_description.GetObject());
	auto creator = return_if_possible<std::string>(
	    "creator", dataset_description.GetObject());
	auto contributor = return_if_possible<std::string>(
	    "contributor", dataset_description.GetObject());
	auto collection_date = return_if_possible<std::string>(
	    "collection_date", dataset_description.GetObject());
	auto upload_date = return_if_possible<std::string>(
	    "upload_date", dataset_description.GetObject());
	auto language = return_if_possible<std::string>(
	    "language", dataset_description.GetObject());
	auto licence = return_if_possible<std::string>(
	    "licence", dataset_description.GetObject());
	auto url =
	    return_if_possible<std::string>("url", dataset_description.GetObject());
	auto default_target_attribute = return_if_possible<std::string>(
	    "default_target_attribute", dataset_description.GetObject());
	auto row_id_attribute = return_if_possible<std::string>(
	    "row_id_attribute", dataset_description.GetObject());
	auto ignore_attribute = return_if_possible<std::string>(
	    "ignore_attribute", dataset_description.GetObject());
	auto version_label = return_if_possible<std::string>(
	    "version_label", dataset_description.GetObject());
	auto citation = return_if_possible<std::string>(
	    "citation", dataset_description.GetObject());
	auto tags = return_if_possible<std::vector<std::string>>(
	    "tag", dataset_description.GetObject());
	auto visibility = return_if_possible<std::string>(
	    "visibility", dataset_description.GetObject());
	auto original_data_url = return_if_possible<std::string>(
	    "original_data_url", dataset_description.GetObject());
	auto paper_url = return_if_possible<std::string>(
	    "paper_url", dataset_description.GetObject());
	auto update_comment = return_if_possible<std::string>(
	    "update_comment", dataset_description.GetObject());
	auto md5_checksum = return_if_possible<std::string>(
	    "md5_checksum", dataset_description.GetObject());

	// features
	std::vector<std::unordered_map<std::string, std::vector<std::string>>>
	    param_vector;
	return_string = reader.get("data_features", "json", id);

	auto& dataset_features =
	    check_response<BACKEND_FORMAT::JSON>(return_string, "data_features");

	for (const auto& param : dataset_features["feature"].GetArray())
	{
		std::unordered_map<std::string, std::vector<std::string>> param_map;
		for (const auto& param_descriptors : param.GetObject())
		{
			std::vector<std::string> second;
			if (param_descriptors.value.IsArray())
				for (const auto& v : param_descriptors.value.GetArray())
					second.emplace_back(v.GetString());
			else
				second.emplace_back(param_descriptors.value.GetString());

			param_map.emplace(param_descriptors.name.GetString(), second);
		}
		param_vector.push_back(param_map);
	}

	// qualities
	std::vector<std::unordered_map<std::string, std::string>> qualities_vector;
	return_string = reader.get("data_qualities", "json", id);

	auto& data_qualities =
	    check_response<BACKEND_FORMAT::JSON>(return_string, "data_qualities");

	for (const auto& param : data_qualities["quality"].GetArray())
	{
		std::unordered_map<std::string, std::string> param_map;
		for (const auto& param_quality : param.GetObject())
		{
			if (param_quality.name.IsString() && param_quality.value.IsString())
				param_map.emplace(
				    param_quality.name.GetString(),
				    param_quality.value.GetString());
			else if (param_quality.name.IsString())
				param_map.emplace(param_quality.name.GetString(), "");
		}
		qualities_vector.push_back(param_map);
	}

	auto result = std::make_shared<OpenMLData>(
	    name, description, data_format, dataset_id, version, creator,
	    contributor, collection_date, upload_date, language, licence, url,
	    default_target_attribute, row_id_attribute, ignore_attribute,
	    version_label, citation, tags, visibility, original_data_url, paper_url,
	    update_comment, md5_checksum, param_vector, qualities_vector);
	result->set_api_key(api_key);
	return result;
}

std::shared_ptr<CFeatures> OpenMLData::get_features() noexcept
{
	//	if (!m_cached_features)
	//		get_data();
	//	return m_cached_features;
	SG_SNOTIMPLEMENTED
	return nullptr;
}

std::shared_ptr<CFeatures> OpenMLData::get_features(const std::string& label)
{
	if (m_cached_features.empty())
		get_data();
	auto find_label =
	    std::find(m_feature_names.begin(), m_feature_names.end(), label);
	if (find_label == m_feature_names.end())
		SG_SERROR("Requested label \"%s\" not in the dataset!\n", label.c_str())
	auto col_idx = std::distance(m_feature_names.begin(), find_label);
	auto feat_type_copy = m_feature_types;
	feat_type_copy.erase(feat_type_copy.begin() + col_idx);
	for (const auto type : feat_type_copy)
	{
		if (type == Attribute::STRING)
			SG_SNOTIMPLEMENTED
	}

	std::shared_ptr<CDenseFeatures<float64_t>> result;
	bool first = true;
	size_t n_examples = 0;
	for (int i = 0; i < m_feature_types.size(); ++i)
	{
		if (i != col_idx && first)
		{
			result.reset(m_cached_features[0]->as<CDenseFeatures<float64_t>>());
			n_examples = result->get_num_vectors();
			first = false;
		}
		if (i != col_idx)
		{
			REQUIRE(
			    n_examples == m_cached_features[i]->get_num_vectors(),
			    "Expected all features to have the same number of examples!\n")
			result.reset(result->create_merged_copy(m_cached_features[i].get())
			                 ->as<CDenseFeatures<float64_t>>());
		}
	}

	REQUIRE(n_examples != 0, "No features extracted!\n")

	// need to copy data as result is only in the stack and the data
	// will be gone at the end of the function
	auto* copy_feat = SG_MALLOC(float64_t, m_feature_types.size() * n_examples);
	memcpy(
	    copy_feat, result->get_feature_matrix().data(),
	    m_feature_types.size() * m_cached_features.size());

	result = std::make_shared<CDenseFeatures<float64_t>>(
	    copy_feat, m_feature_types.size(), n_examples);

	return result;
}

std::shared_ptr<CLabels> OpenMLData::get_labels()
{
	REQUIRE(
	    !m_default_target_attribute.empty(),
	    "A default target attribute is required if no label is given!\n")
	return get_labels(m_default_target_attribute);
}

std::shared_ptr<CLabels> OpenMLData::get_labels(const std::string& label_name)
{
	if (m_cached_labels && label_name == m_cached_label_name)
		return m_cached_labels;

	if (m_cached_features.empty())
		get_data();

	auto find_label =
	    std::find(m_feature_names.begin(), m_feature_names.end(), label_name);
	if (find_label == m_feature_names.end())
		SG_SERROR(
		    "Requested label \"%s\" not in the dataset!\n", label_name.c_str())
	auto col_idx = std::distance(m_feature_names.begin(), find_label);

	std::shared_ptr<CFeatures> target_label_as_feat =
	    m_cached_features[col_idx];

	switch (m_feature_types[col_idx])
	{
	// real features
	case Attribute::REAL:
	case Attribute::NUMERIC:
	case Attribute::INTEGER:
	case Attribute::DATE:
	{
		auto casted_feat = std::dynamic_pointer_cast<CDenseFeatures<float64_t>>(
		    target_label_as_feat);
		auto labels_vec = casted_feat->get_feature_matrix().get_row_vector(0);
		auto labels = std::make_shared<CRegressionLabels>(labels_vec);
		m_cached_labels = labels;
		m_cached_label_name = label_name;
		return m_cached_labels;
	}
	break;
		// nominal features
	case Attribute::NOMINAL:
	{
		auto casted_feat = std::dynamic_pointer_cast<CDenseFeatures<float64_t>>(
		    target_label_as_feat);
		auto labels_vec = casted_feat->get_feature_matrix().get_row_vector(0);
		for (auto& val : labels_vec)
		{
			if (val == 0)
				val = -1;
		}
		auto labels = std::make_shared<CBinaryLabels>(labels_vec);
		m_cached_labels = labels;
		m_cached_label_name = label_name;
		return m_cached_labels;
	}
	break;
	default:
		SG_SERROR("Unknown type for label \"%s\"!\n", label_name.c_str())
	}

	return nullptr;
}

void OpenMLData::get_data()
{
	auto reader = OpenMLFile(m_api_key);
	std::shared_ptr<std::istream> ss =
	    std::make_shared<std::istringstream>(reader.get(m_url));

	auto parser = ARFFDeserializer(ss);
	parser.read();
	m_cached_features = parser.get_features();
	m_feature_names = parser.get_feature_names();
	m_feature_types = parser.get_attribute_types();
}
