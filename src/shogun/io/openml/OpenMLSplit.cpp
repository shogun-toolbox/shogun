/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/features/CombinedFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/StringFeatures.h>

#include <shogun/io/openml/OpenMLReader.h>
#include <shogun/io/openml/OpenMLSplit.h>

using namespace shogun;

std::shared_ptr<OpenMLSplit>
OpenMLSplit::get_split(const std::string& split_url, const std::string& api_key)
{
	auto reader = OpenMLReader(api_key);
	auto return_string = reader.get("get_split", "split", split_url);

	if (return_string == "Task not providing datasplits.")
		return std::make_shared<OpenMLSplit>();

	auto return_stream = std::istringstream(return_string);
	// TODO: add ARFF parsing here
	// get train/test indices
	// TODO: replace line below with ARFFDeserialiser::get_features()
	auto arff_features = std::make_shared<CCombinedFeatures>();
	REQUIRE(
	    arff_features->get_num_feature_obj() == 4,
	    "Expected a ARFF file with 4 attributes: type, rowid, repeat and "
	    "fold.\n")

	auto train_test_feat =
	    std::shared_ptr<CFeatures>(arff_features->get_feature_obj(0));
	auto rowid_feat =
	    std::shared_ptr<CFeatures>(arff_features->get_feature_obj(1));
	auto repeat_feat =
	    std::shared_ptr<CFeatures>(arff_features->get_feature_obj(2));
	auto fold_feat =
	    std::shared_ptr<CFeatures>(arff_features->get_feature_obj(3));

	auto type_vector = string_feature_to_vector(train_test_feat);
	auto rowid_vector = dense_feature_to_vector(rowid_feat);
	auto repeat_vector = dense_feature_to_vector(repeat_feat);
	auto fold_vector = dense_feature_to_vector(fold_feat);

	std::vector<std::vector<int64_t>> train_idx, test_idx;
	for (int i = 0; i < arff_features->get_num_vectors(); ++i)
	{
		if (type_vector[i] == LabelType::TRAIN)
			train_idx.emplace_back(std::initializer_list<int64_t>{
			    static_cast<int64_t>(rowid_vector[i]),
			    static_cast<int64_t>(repeat_vector[i]),
			    static_cast<int64_t>(fold_vector[i])});
		else
			test_idx.emplace_back(std::initializer_list<int64_t>{
			    static_cast<int64_t>(rowid_vector[i]),
			    static_cast<int64_t>(repeat_vector[i]),
			    static_cast<int64_t>(fold_vector[i])});
	}

	return std::make_shared<OpenMLSplit>(train_idx, test_idx);
}

SGVector<float64_t>
OpenMLSplit::dense_feature_to_vector(const std::shared_ptr<CFeatures>& feat)
{
	auto casted_feat =
	    std::dynamic_pointer_cast<CDenseFeatures<float64_t>>(feat);
	// this should never happen
	if (!casted_feat)
		SG_SERROR("Error casting a column in the split file from CFeatures to "
		          "CDenseFeatures!\n>");
	return casted_feat->get_feature_vector(0);
}

std::vector<OpenMLSplit::LabelType>
OpenMLSplit::string_feature_to_vector(const std::shared_ptr<CFeatures>& feat)
{
	auto casted_feat = std::dynamic_pointer_cast<CStringFeatures<char>>(feat);
	// this should never happen
	if (!casted_feat)
		SG_SERROR("Error casting a column in the split file from CFeatures to "
		          "CStringFeatures!\n");

	auto to_lower = [](const std::string& line) {
		std::string result;
		std::transform(
		    line.begin(), line.end(), std::back_inserter(result),
		    [](uint8_t val) { return std::tolower(val); });
		return result;
	};

	std::vector<OpenMLSplit::LabelType> result;

	for (int i = 0; i < casted_feat->get_num_vectors(); ++i)
	{
		auto row = casted_feat->get_feature_vector(i);
		std::string label(1, row[0]);
		for (auto j = 1; j < casted_feat->get_max_vector_length(); ++j)
			label.append(1, row[j]);
		if (to_lower(label) == "train")
			result.push_back(LabelType::TRAIN);
		else if (to_lower(label) == "test")
			result.push_back(LabelType::TEST);
		else
			SG_SERROR("Unknown label type in split file %s!\n", label.c_str())
	}
	return result;
}