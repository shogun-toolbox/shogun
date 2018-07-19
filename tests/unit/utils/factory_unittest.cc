#include "utils/Utils.h"
#include <gtest/gtest.h>
#include <shogun/base/class_list.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/util/factory.h>

using namespace shogun;

class GaussianKernel;

TEST(Factory, kernel)
{
	auto* obj = kernel("GaussianKernel");
	EXPECT_TRUE(obj != nullptr);
	EXPECT_TRUE(dynamic_cast<CGaussianKernel*>(obj) != nullptr);
	delete obj;
}

TEST(Factory, machine)
{
	auto* obj = machine("LibSVM");
	EXPECT_TRUE(obj != nullptr);
	EXPECT_TRUE(dynamic_cast<CLibSVM*>(obj) != nullptr);
	delete obj;
}

TEST(Factory, features_from_matrix)
{
	SGMatrix<float64_t> mat(2, 3);
	auto* obj = features(mat);
	EXPECT_TRUE(obj != nullptr);
	EXPECT_TRUE(dynamic_cast<CDenseFeatures<float64_t>*>(obj) != nullptr);
	delete obj;
}

TEST(Factory, string_features_from_file)
{
	const std::string strings[] = {"GAGACGGACC", "GCGCATATT", "GCGCATATG"};
	std::string filename = "Factory-features_from_file.XXXXXX";

	auto num_strings = 3;
	auto max_string_length = 10;

	SGStringList<char> string_list(num_strings, max_string_length);

	for (auto i : range(3))
	{
		string_list.strings[i].string = const_cast<char*>(strings[i].c_str());
		string_list.strings[i].slen = strings[i].size();
	}

	generate_temp_filename(const_cast<char*>(filename.c_str()));
	auto file_save = some<CCSVFile>(filename.c_str(), 'w');
	string_list.save(file_save);
	file_save->close();

	auto file_load = some<CCSVFile>(filename.c_str(), 'r');
	auto obj = wrap(string_features(file_load, DNA, PT_CHAR));
	file_load->close();

	EXPECT_TRUE(obj.get() != nullptr);
	auto cast = obj->as<CStringFeatures<char>>();
	auto loaded_string_list = cast->get_features();

	EXPECT_EQ(loaded_string_list.num_strings, string_list.num_strings);
	for (auto i : range(loaded_string_list.num_strings))
	{
		EXPECT_TRUE(
		    loaded_string_list.strings[i].equals(string_list.strings[i]));
	}

	EXPECT_EQ(std::remove(filename.c_str()), 0);
}

template<class T>
void test_label_factory_spawns_correct_type(SGVector<float64_t>& vec,
		bool expects_fail=false)
{
	std::string filename = "Factory-labels_from_file.XXXXXX";
	generate_temp_filename(const_cast<char*>(filename.c_str()));
	auto file_save = some<CCSVFile>(filename.c_str(), 'w');
	vec.save(file_save);
	file_save->close();

	auto file_load = some<CCSVFile>(filename.c_str(), 'r');
	auto* obj = labels(file_load);
	file_load->close();

	EXPECT_TRUE(obj != nullptr);
	auto* cast = dynamic_cast<T*>(obj);

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

	delete obj;
	EXPECT_EQ(std::remove(filename.c_str()), 0);
}

TEST(Factory, labels_binary_from_file)
{
	SGVector<float64_t> vec;

	vec = {-1, -1, 1, 1};
	test_label_factory_spawns_correct_type<CBinaryLabels>(vec);

	vec = {-1};
	test_label_factory_spawns_correct_type<CBinaryLabels>(vec);

	vec = {1, 1};
	test_label_factory_spawns_correct_type<CBinaryLabels>(vec);

	vec = {0, 1};
	test_label_factory_spawns_correct_type<CBinaryLabels>(vec, true);
}

TEST(Factory, labels_multiclass_from_file)
{
	SGVector<float64_t> vec;

	vec = {0, 1, 2};
	test_label_factory_spawns_correct_type<CMulticlassLabels>(vec);

	vec = {0};
	test_label_factory_spawns_correct_type<CMulticlassLabels>(vec);

	vec = {1, 1, 3, 4};
	test_label_factory_spawns_correct_type<CMulticlassLabels>(vec);

	// should create binary
	vec = {1, 1};
	test_label_factory_spawns_correct_type<CMulticlassLabels>(vec, true);
	vec = {1, -1};
	test_label_factory_spawns_correct_type<CMulticlassLabels>(vec, true);

	// should create regression
	vec = {1, 1.1};
	test_label_factory_spawns_correct_type<CMulticlassLabels>(vec, true);
}

TEST(Factory, labels_regression_from_file)
{
	SGVector<float64_t> vec;

	vec = {1.0, 1.1};
	test_label_factory_spawns_correct_type<CRegressionLabels>(vec);

	// should create binary
	vec = {-1, 1};
	test_label_factory_spawns_correct_type<CRegressionLabels>(vec, true);
	vec = {1, 1};
	test_label_factory_spawns_correct_type<CRegressionLabels>(vec, true);
	vec = {-1, -1};
	test_label_factory_spawns_correct_type<CRegressionLabels>(vec, true);

	// should create multiclass
	vec = {0, 1};
	test_label_factory_spawns_correct_type<CRegressionLabels>(vec, true);
}
