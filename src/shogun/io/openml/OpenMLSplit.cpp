/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/features/CombinedFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/StringFeatures.h>

#include <shogun/io/ARFFFile.h>
#include <shogun/io/openml/OpenMLFile.h>
#include <shogun/io/openml/OpenMLSplit.h>

using namespace shogun;

std::shared_ptr<OpenMLSplit>
OpenMLSplit::get_split(const std::string& split_url, const std::string& api_key)
{
	auto reader = OpenMLFile(api_key);
	auto return_string = reader.get(split_url);

	if (return_string == "Task not providing datasplits.")
		return std::make_shared<OpenMLSplit>();

	std::shared_ptr<std::istream> return_stream =
	    std::make_shared<std::istringstream>(return_string);
	auto arff_parser = ARFFDeserializer(return_stream);
	arff_parser.read();
	auto arff_features = arff_parser.get_features();
	REQUIRE(
	    arff_features.size() == 4,
	    "Expected a ARFF file with 4 attributes: type, rowid, repeat and "
	    "fold.\n")

	auto type_vector = nominal_feature_to_vector(arff_features[0]);
	auto rowid_vector = dense_feature_to_vector(arff_features[1]);
	auto repeat_vector = dense_feature_to_vector(arff_features[2]);
	auto fold_vector = dense_feature_to_vector(arff_features[3]);

	std::array<std::vector<int32_t>, 3> train_idx, test_idx;

	for (int i = 0; i < arff_features[0]->get_num_vectors(); ++i)
	{
		if (type_vector[i] == LabelType::TRAIN)
		{
			train_idx[0].push_back(rowid_vector[i]);
			train_idx[1].push_back(repeat_vector[i]);
			train_idx[2].push_back(fold_vector[i]);
		}
		else
		{
			test_idx[0].push_back(rowid_vector[i]);
			test_idx[1].push_back(repeat_vector[i]);
			test_idx[2].push_back(fold_vector[i]);
		}
	}

	return std::make_shared<OpenMLSplit>(train_idx, test_idx);
}

SGMatrix<float64_t>
OpenMLSplit::dense_feature_to_vector(const std::shared_ptr<CFeatures>& feat)
{
	auto casted_feat =
	    std::dynamic_pointer_cast<CDenseFeatures<float64_t>>(feat);
	// this should never happen
	if (!casted_feat)
		SG_SERROR("Error casting a column in the split file from CFeatures to "
		          "CDenseFeatures!\n>");
	return casted_feat->get_feature_matrix();
}

std::vector<OpenMLSplit::LabelType>
OpenMLSplit::nominal_feature_to_vector(const std::shared_ptr<CFeatures>& feat)
{
	auto casted_feat =
	    std::dynamic_pointer_cast<CDenseFeatures<float64_t>>(feat);
	// this should never happen
	if (!casted_feat)
		SG_SERROR("Error casting a column in the split file from CFeatures to "
		          "CDenseFeatures!\n");

	std::vector<OpenMLSplit::LabelType> result;

	for (const auto& el : casted_feat->get_feature_matrix())
	{
		if (el == 0)
			result.push_back(LabelType::TRAIN);
		else
			result.push_back(LabelType::TEST);
	}

	return result;
}