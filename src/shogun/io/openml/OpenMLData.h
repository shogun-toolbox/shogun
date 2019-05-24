/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_OPENMLDATA_H
#define SHOGUN_OPENMLDATA_H

#include <shogun/features/CombinedFeatures.h>
#include <shogun/io/ARFFFile.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace shogun
{
	/**
	 * Handles an OpenML dataset.
	 */
	class OpenMLData
	{
	public:
		OpenMLData(
		    const std::string& name, const std::string& description,
		    const std::string& data_format, const std::string& dataset_id,
		    const std::string& version, const std::string& creator,
		    const std::string& contributor, const std::string& collection_date,
		    const std::string& upload_date, const std::string& language,
		    const std::string& license, const std::string& url,
		    const std::string& default_target_attribute,
		    const std::string& row_id_attribute,
		    const std::string& ignore_attribute,
		    const std::string& version_label, const std::string& citation,
		    std::vector<std::string> tag, const std::string& visibility,
		    const std::string& original_data_url, const std::string& paper_url,
		    const std::string& update_comment, const std::string& md5_checksum,
		    std::vector<
		        std::unordered_map<std::string, std::vector<std::string>>

		        >
		        param_descriptors,
		    std::vector<std::unordered_map<std::string, std::string>>
		        param_qualities)
		    :

		      m_name(name), m_description(description),
		      m_data_format(data_format), m_dataset_id(dataset_id),
		      m_version(version), m_creator(creator),
		      m_contributor(contributor), m_collection_date(collection_date),
		      m_upload_date(upload_date), m_language(language),
		      m_license(license), m_url(url),
		      m_default_target_attribute(default_target_attribute),
		      m_row_id_attribute(row_id_attribute),
		      m_ignore_attribute(ignore_attribute),
		      m_version_label(version_label), m_citation(citation),
		      m_tag(std::move(tag)), m_visibility(visibility),
		      m_original_data_url(original_data_url), m_paper_url(paper_url),
		      m_update_comment(update_comment), m_md5_checksum(md5_checksum),
		      m_param_descriptors(std::move(param_descriptors)),
		      m_param_qualities(std::move(param_qualities))
		{
		}

		/**
		 * Creates a dataset instance from a given ID.
		 *
		 */
		static std::shared_ptr<OpenMLData>
		get_dataset(const std::string& id, const std::string& api_key);

		/**
		 * Returns ALL the features of the dataset, potentially also the labels
		 * column
		 * @return the features
		 */
		std::shared_ptr<CFeatures> get_features() noexcept;

		/**
		 * Returns the dataset features
		 * @param label_name the name of the attribute containing the label
		 * @return the features
		 */
		std::shared_ptr<CFeatures> get_features(const std::string& label_name);

		/**
		 * Returns the dataset labels if m_default_target_attribute is not empty
		 * @return the labels
		 */
		std::shared_ptr<CLabels> get_labels();

		/**
		 * Returns the dataset labels given the label_name
		 * @return the labels
		 */
		std::shared_ptr<CLabels> get_labels(const std::string& label_name);

		/**
		 * Returns the type of all attributes/features in the ARFF file
		 * @return
		 */
		SG_FORCED_INLINE std::vector<Attribute> get_feature_types() const
		    noexcept
		{
			return m_feature_types;
		}

		SG_FORCED_INLINE std::string get_default_target_attribute() const
		    noexcept
		{
			return m_default_target_attribute;
		}

	protected:
		SG_FORCED_INLINE void set_api_key(const std::string& api_key) noexcept
		{
			m_api_key = api_key;
		}

	private:
		void get_data();

		std::string m_name;
		std::string m_description;
		std::string m_data_format;
		std::string m_dataset_id;
		std::string m_version;
		std::string m_creator;
		std::string m_contributor;
		std::string m_collection_date;
		std::string m_upload_date;
		std::string m_language;
		std::string m_license;
		std::string m_url;
		std::string m_default_target_attribute;
		std::string m_row_id_attribute;
		std::string m_ignore_attribute;
		std::string m_version_label;
		std::string m_citation;
		std::vector<std::string> m_tag;
		std::string m_visibility;
		std::string m_original_data_url;
		std::string m_paper_url;
		std::string m_update_comment;
		std::string m_md5_checksum;
		std::vector<std::unordered_map<std::string, std::vector<std::string>>>
		    m_param_descriptors;
		std::vector<std::unordered_map<std::string, std::string>>
		    m_param_qualities;
		std::string m_api_key;

		std::vector<std::shared_ptr<CFeatures>> m_cached_features;
		std::vector<std::string> m_feature_names;
		std::vector<Attribute> m_feature_types;
		std::shared_ptr<CLabels> m_cached_labels;
		std::string m_cached_label_name;
	};

} // namespace shogun

#endif // SHOGUN_OPENMLDATA_H
