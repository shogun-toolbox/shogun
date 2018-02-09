/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */
#include <gtest/gtest.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/labels/BinaryLabels.h>

#include "utils/Utils.h"

using namespace shogun;

class BinaryLabels : public ::testing::Test
{
public:
	SGVector<float64_t> probabilities;
	const int32_t n_A = 4;

	virtual void SetUp()
	{
		auto A = SGVector<float64_t>(n_A);
		A[0] = 0.1;
		A[1] = 0.4;
		A[2] = 0.6;
		A[3] = 0.9;
		probabilities = A;
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
	const float64_t threshold = 0.5;
	CBinaryLabels* labels = new CBinaryLabels(probabilities, threshold);

	SGVector<float64_t> labels_vector = labels->get_labels();
	SGVector<float64_t> values_vector = labels->get_values();

	ASSERT(labels_vector);
	ASSERT(values_vector);

	ASSERT_EQ(n_A, labels_vector.size());
	ASSERT_EQ(n_A, values_vector.size());

	EXPECT_FLOAT_EQ(-1.0, labels_vector[0]);
	EXPECT_FLOAT_EQ(-1.0, labels_vector[1]);
	EXPECT_FLOAT_EQ(+1.0, labels_vector[2]);
	EXPECT_FLOAT_EQ(+1.0, labels_vector[3]);

	for (int i = 0; i < values_vector.size(); ++i)
	{
		EXPECT_FLOAT_EQ(probabilities[i], values_vector[i]);
	}

	SG_UNREF(labels);
}
