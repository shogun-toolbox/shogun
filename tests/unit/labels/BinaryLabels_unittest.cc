/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */
#include "utils/Utils.h"
#include <gtest/gtest.h>
#include <shogun/base/range.h>
#include <shogun/io/SerializableAsciiFile.h>
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
	CBinaryLabels* labels = new CBinaryLabels(10);
	SGVector<float64_t> lab = SGVector<float64_t>(labels->get_num_labels());
	lab.random(1, 10);
	labels->set_values(lab);
	labels->set_labels(lab);

	/* generate file name */
	char filename[] = "serialization-asciiCBinaryLabels.XXXXXX";
	generate_temp_filename(filename);

	CSerializableAsciiFile* file = new CSerializableAsciiFile(filename, 'w');
	labels->save_serializable(file);
	file->close();
	SG_UNREF(file);

	file = new CSerializableAsciiFile(filename, 'r');
	CBinaryLabels* new_labels = new CBinaryLabels;
	new_labels->load_serializable(file);
	file->close();
	SG_UNREF(file);

	ASSERT(new_labels->get_num_labels() == 10)

	for (int32_t i = 0; i < new_labels->get_num_labels(); i++)
	{
		EXPECT_NEAR(labels->get_value(i), new_labels->get_value(i), 1E-15);
		EXPECT_NEAR(labels->get_label(i), new_labels->get_label(i), 1E-15);
	}
	unlink(filename);

	SG_UNREF(labels);
	SG_UNREF(new_labels);
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

TEST_F(BinaryLabels, binary_labels_from_dense)
{
	auto labels = some<CDenseLabels>(probabilities.size());
	labels->set_values(probabilities);
	labels->set_labels(labels_binary);

	auto labels2 = binary_labels(labels);
	EXPECT_NE(labels, labels2);
	ASSERT_NE(labels2, nullptr);
	EXPECT_EQ(labels->get_labels(), labels2->get_labels());
	EXPECT_EQ(labels->get_values(), labels2->get_values());
}
