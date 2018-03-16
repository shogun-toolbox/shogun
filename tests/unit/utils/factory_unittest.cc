#include "utils/Utils.h"
#include <shogun/base/class_list.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/util/factory.h>

#include <gtest/gtest.h>

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

TEST(Factory, features_dense_from_file)
{
	std::string filename = "Factory-features_from_file.XXXXXX";

	SGMatrix<float64_t> mat(2, 3);
	mat.set_const(1);
	generate_temp_filename(const_cast<char*>(filename.c_str()));
	auto file_save = some<CCSVFile>(filename.c_str(), 'w');
	mat.save(file_save);
	file_save->close();

	auto file_load = some<CCSVFile>(filename.c_str(), 'r');
	auto* obj = features(file_load, PT_FLOAT64);
	file_load->close();
	EXPECT_TRUE(obj != nullptr);
	auto* cast = dynamic_cast<CDenseFeatures<float64_t>*>(obj);
	ASSERT_TRUE(cast != nullptr);
	auto loaded_mat = cast->get_feature_matrix();
	EXPECT_TRUE(loaded_mat.equals(mat));
	delete obj;

	EXPECT_EQ(std::remove(filename.c_str()), 0);
}

TEST(Factory, labels_from_file)
{
	std::string filename = "Factory-labels_from_file.XXXXXX";

	SGVector<float64_t> vec(3);
	vec.set_const(1);
	generate_temp_filename(const_cast<char*>(filename.c_str()));
	auto file_save = some<CCSVFile>(filename.c_str(), 'w');
	vec.save(file_save);
	file_save->close();

	auto file_load = some<CCSVFile>(filename.c_str(), 'r');
	auto* obj = labels(file_load);
	file_load->close();
	EXPECT_TRUE(obj != nullptr);
	auto* cast = dynamic_cast<CDenseLabels*>(obj);
	ASSERT_TRUE(cast != nullptr);
	auto loaded_vec = cast->get_labels();
	EXPECT_TRUE(loaded_vec.equals(vec));
	delete obj;

	EXPECT_EQ(std::remove(filename.c_str()), 0);
}
