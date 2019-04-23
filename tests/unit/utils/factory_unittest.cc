#include "utils/Utils.h"
#include <gtest/gtest.h>
#include <shogun/base/class_list.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/util/factory.h>

using namespace shogun;

TEST(Factory, kernel)
{
	auto obj = kernel("GaussianKernel");
	EXPECT_TRUE(obj != nullptr);
	EXPECT_TRUE(obj->as<GaussianKernel>() != nullptr);
}

TEST(Factory, machine)
{
	auto obj = machine("LibSVM");
	EXPECT_TRUE(obj != nullptr);
	EXPECT_TRUE(obj->as<LibSVM>() != nullptr);
}

TEST(Factory, features_from_matrix)
{
	SGMatrix<float64_t> mat(2, 3);
	auto obj = features(mat);
	EXPECT_TRUE(obj != nullptr);
	EXPECT_TRUE(obj->as<DenseFeatures<float64_t>>() != nullptr);
}

// FIXME
TEST(Factory, DISABLED_string_features_from_file)
{
	const std::string strings[] = {"GAGACGGACC", "GCGCATATT", "GCGCATATG"};
	std::string filename = "Factory-features_from_file.XXXXXX";

	auto num_strings = 3;

	std::vector<SGVector<char>> string_list;
	string_list.reserve(num_strings);

	for (auto i : range(3))
	{
		string_list.emplace_back(const_cast<char*>(strings[i].c_str()), strings[i].size(), false);
	}

	generate_temp_filename(const_cast<char*>(filename.c_str()));
	auto file_save = std::make_shared<CSVFile>(filename.c_str(), 'w');
	SG_SET_LOCALE_C;
	file_save->set_string_list(string_list.data(), string_list.size());
	SG_RESET_LOCALE;
	file_save->close();

	auto file_load = std::make_shared<CSVFile>(filename.c_str(), 'r');
	auto obj = string_features(file_load, DNA, PT_CHAR);
	file_load->close();

	EXPECT_TRUE(obj.get() != nullptr);
	auto cast = obj->as<StringFeatures<char>>();
	auto loaded_string_list = cast->get_string_list();

	EXPECT_EQ(loaded_string_list.size(), string_list.size());
	for (auto i : range(loaded_string_list.size()))
	{
		EXPECT_TRUE(
		    loaded_string_list[i].equals(string_list[i]));
	}

	EXPECT_EQ(std::remove(filename.c_str()), 0);
}

//fixme
#if 0
template <class T>
void test_label_factory_spawns_correct_type(
    SGVector<float64_t>& vec, bool expects_fail = false)
{
	std::string filename = "Factory-labels_from_file.XXXXXX";
	generate_temp_filename(const_cast<char*>(filename.c_str()));
	auto file_save = std::make_shared<CSVFile>(filename.c_str(), 'w');
	vec.save(file_save.get());
	file_save->close();

	auto file_load = std::make_shared<CSVFile>(filename.c_str(), 'r');
	std::shared_ptr<Labels> obj= labels(file_load);
	file_load->close();

	EXPECT_TRUE(obj != nullptr);
	auto cast = obj->as<T>();

	if (expects_fail)
	{
		EXPECT_TRUE(cast == nullptr);
	}
	else
	{
		ASSERT_TRUE(cast != nullptr);
		auto loaded_vec = cast->get_labels();
		EXPECT_TRUE(loaded_vec.equals(vec));
	}

	EXPECT_EQ(std::remove(filename.c_str()), 0);
}


TEST(Factory, DISABLED_labels_binary_from_file)
{
	SGVector<float64_t> vec;

	vec = {-1, -1, 1, 1};
	test_label_factory_spawns_correct_type<BinaryLabels>(vec);

	vec = {-1};
	test_label_factory_spawns_correct_type<BinaryLabels>(vec);

	vec = {1, 1};
	test_label_factory_spawns_correct_type<BinaryLabels>(vec);

	vec = {0, 1};
	test_label_factory_spawns_correct_type<BinaryLabels>(vec, true);
}

//fixme
TEST(Factory, DISABLED_labels_multiclass_from_file)
{
	SGVector<float64_t> vec;

	vec = {0, 1, 2};
	test_label_factory_spawns_correct_type<MulticlassLabels>(vec);

	vec = {0};
	test_label_factory_spawns_correct_type<MulticlassLabels>(vec);

	vec = {1, 1, 3, 4};
	test_label_factory_spawns_correct_type<MulticlassLabels>(vec);

	// should create binary
	vec = {1, 1};
	test_label_factory_spawns_correct_type<MulticlassLabels>(vec, true);
	vec = {1, -1};
	test_label_factory_spawns_correct_type<MulticlassLabels>(vec, true);

	// should create regression
	vec = {1, 1.1};
	test_label_factory_spawns_correct_type<MulticlassLabels>(vec, true);
}

//fixme
TEST(Factory, DISABLED_labels_regression_from_file)
{
	SGVector<float64_t> vec;

	vec = {1.0, 1.1};
	test_label_factory_spawns_correct_type<RegressionLabels>(vec);

	// should create binary
	vec = {-1, 1};
	test_label_factory_spawns_correct_type<RegressionLabels>(vec, true);
	vec = {1, 1};
	test_label_factory_spawns_correct_type<RegressionLabels>(vec, true);
	vec = {-1, -1};
	test_label_factory_spawns_correct_type<RegressionLabels>(vec, true);

	// should create multiclass
	vec = {0, 1};
	test_label_factory_spawns_correct_type<RegressionLabels>(vec, true);
}
#endif
