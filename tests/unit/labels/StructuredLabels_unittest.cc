/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Fernando Iglesias, Akash Shivram
 */

#include <shogun/labels/StructuredLabels.h>
#include <shogun/structure/MulticlassSOLabels.h>
#include <gtest/gtest.h>

using namespace shogun;


TEST(StructuredLabels, add_label)
{
	int32_t num_labels = 3;
	auto l = std::make_shared<StructuredLabels>(num_labels);

	l->add_label(std::make_shared<RealNumber>(3));
	l->add_label(std::make_shared<RealNumber>(7));
	l->add_label(std::make_shared<RealNumber>(13));

	std::shared_ptr<RealNumber> real_number;

	EXPECT_EQ(3, l->get_num_labels());

	real_number = l->get_label(0)->as<RealNumber>();
	EXPECT_EQ(3, real_number->value);


	real_number = l->get_label(1)->as<RealNumber>();
	EXPECT_EQ(7, real_number->value);


	real_number = l->get_label(2)->as<RealNumber>();
	EXPECT_EQ(13, real_number->value);



}

TEST(StructuredLabels, set_label)
{
	int32_t num_labels = 3;
	auto l = std::make_shared<StructuredLabels>(num_labels);

	l->add_label(std::make_shared<RealNumber>(3));
	l->add_label(std::make_shared<RealNumber>(7));
	l->add_label(std::make_shared<RealNumber>(13));

	l->set_label(1, std::make_shared<RealNumber>(23));

	std::shared_ptr<RealNumber> real_number;

	EXPECT_EQ(3, l->get_num_labels());

	real_number = l->get_label(0)->as<RealNumber>();
	EXPECT_EQ(3, real_number->value);


	real_number = l->get_label(1)->as<RealNumber>();
	EXPECT_EQ(23, real_number->value);


	real_number = l->get_label(2)->as<RealNumber>();
	EXPECT_EQ(13, real_number->value);



}
