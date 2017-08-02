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

	virtual void SetUp()
	{
		auto A = SGVector<float64_t>(4);
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

TEST_F(BinaryLabels, scores_to_probabilities)
{
	CBinaryLabels* labels=new CBinaryLabels(10);
	labels->set_values(SGVector<float64_t>(labels->get_num_labels()));

	for (index_t i=0; i<labels->get_num_labels(); ++i)
		labels->set_value(i%2==0 ? 1 : -1, i);

	//labels->get_values().display_vector("scores");
	// call with 0,0 to make the method compute sigmoid parameters itself
	// g-test somehow does not allow std parameters
	labels->scores_to_probabilities(0,0);

	/* only two probabilities will be the result. Results from implementation that
	 * comes with the original paper, see BinaryLabels documentation */
	EXPECT_NEAR(labels->get_value(0), 0.8571428439385661, 10E-15);
	EXPECT_NEAR(labels->get_value(1), 0.14285715606143384, 10E-15);

	SG_UNREF(labels);
}

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
	CBinaryLabels* labels = new CBinaryLabels(probabilities);

	SGVector<float64_t> labels_vector = labels->get_labels();
	SGVector<float64_t> values_vector = labels->get_values();

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
