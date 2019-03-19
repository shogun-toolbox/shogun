/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, durovo, Olivier NGuyen, Viktor Gal, Weijie Lin,
 *          Thoralf Klein
 */
#include "utils/Utils.h"
#include <gtest/gtest.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/serialization/JsonSerializer.h>
#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/io/stream/FileInputStream.h>
#include <shogun/io/stream/FileOutputStream.h>
#include <shogun/base/range.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

class BinaryLabels : public ::testing::Test
{
public:
	SGVector<float64_t> probabilities;
	SGVector<float64_t> labels_binary;

	const int32_t n = 4;
	float64_t threshold = 0.5;

	virtual void SetUp()
	{
		probabilities = {0.1, 0.4, 06, 0.9};

		auto t = threshold;
		labels_binary = SGVector<float64_t>(n);
		std::transform(
		    probabilities.begin(), probabilities.end(), labels_binary.begin(),
		    [t](float64_t a) { return a > t ? 1 : -1; });
	}

	virtual void TearDown()
	{
	}
};

TEST_F(BinaryLabels, serialization)
{
	auto labels = some<CBinaryLabels>(10);
	SGVector<float64_t> lab = SGVector<float64_t>(labels->get_num_labels());
	lab.random(1, 10);
	labels->set_values(lab);
	labels->set_labels(lab);

	/* generate file name */
	std::string filename = "serialization-json-CBinaryLabels.XXXXXX";
	generate_temp_filename(const_cast<char*>(filename.c_str()));

	auto fs = io::FileSystemRegistry::instance();
	ASSERT_TRUE(fs->file_exists(filename));
	std::unique_ptr<io::WritableFile> file;
	ASSERT_FALSE(fs->new_writable_file(filename, &file));
	auto fos = some<io::CFileOutputStream>(file.get());
	auto serializer = some<io::CJsonSerializer>();
	serializer->attach(fos);
	serializer->write(labels);

	std::unique_ptr<io::RandomAccessFile> raf;
	ASSERT_FALSE(fs->new_random_access_file(filename, &raf));
	auto fis = some<io::CFileInputStream>(raf.get());
	auto deserializer = some<io::CJsonDeserializer>();
	deserializer->attach(fis);
	auto deser_obj = deserializer->read();
	ASSERT_FALSE(fs->delete_file(filename));

	auto new_labels = static_cast<CBinaryLabels*>(deser_obj.get());
	ASSERT_TRUE(new_labels->get_num_labels() == 10)

	for (int32_t i = 0; i < new_labels->get_num_labels(); i++)
	{
		EXPECT_NEAR(labels->get_value(i), new_labels->get_value(i), 1E-15);
		EXPECT_NEAR(labels->get_label(i), new_labels->get_label(i), 1E-15);
	}
}

TEST_F(BinaryLabels, set_values_labels_from_constructor)
{
	CBinaryLabels* labels = new CBinaryLabels(probabilities, threshold);

	SGVector<float64_t> labels_vector = labels->get_labels();
	SGVector<float64_t> values_vector = labels->get_values();

	ASSERT(labels_vector);
	ASSERT(values_vector);

	ASSERT_EQ(n, labels_vector.size());
	ASSERT_EQ(n, values_vector.size());

	for (auto i : range(n))
	{
		EXPECT_FLOAT_EQ(labels_binary[i], labels_vector[i]);
	}

	for (int i = 0; i < values_vector.size(); ++i)
	{
		EXPECT_FLOAT_EQ(probabilities[i], values_vector[i]);
	}

	SG_UNREF(labels);
}

TEST_F(BinaryLabels, binary_labels_from_binary)
{
	auto labels = some<CBinaryLabels>(probabilities, 0.5);
	auto labels2 = binary_labels(labels);
	EXPECT_EQ(labels, labels2);
}
