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
	CStructuredLabels * l = new CStructuredLabels(num_labels);

	l->add_label(new CRealNumber(3));
	l->add_label(new CRealNumber(7));
	l->add_label(new CRealNumber(13));

	CRealNumber* real_number;

	EXPECT_EQ(3, l->get_num_labels());

	real_number = l->get_label(0)->as<CRealNumber>();
	EXPECT_EQ(3, real_number->value);
	SG_UNREF(real_number);

	real_number = l->get_label(1)->as<CRealNumber>();
	EXPECT_EQ(7, real_number->value);
	SG_UNREF(real_number);

	real_number = l->get_label(2)->as<CRealNumber>();
	EXPECT_EQ(13, real_number->value);
	SG_UNREF(real_number);

	SG_UNREF(l);
}

TEST(StructuredLabels, set_label)
{
	int32_t num_labels = 3;
	CStructuredLabels * l = new CStructuredLabels(num_labels);

	l->add_label(new CRealNumber(3));
	l->add_label(new CRealNumber(7));
	l->add_label(new CRealNumber(13));

	l->set_label(1, new CRealNumber(23));

	CRealNumber* real_number;

	EXPECT_EQ(3, l->get_num_labels());

	real_number = l->get_label(0)->as<CRealNumber>();
	EXPECT_EQ(3, real_number->value);
	SG_UNREF(real_number);

	real_number = l->get_label(1)->as<CRealNumber>();
	EXPECT_EQ(23, real_number->value);
	SG_UNREF(real_number);

	real_number = l->get_label(2)->as<CRealNumber>();
	EXPECT_EQ(13, real_number->value);
	SG_UNREF(real_number);

	SG_UNREF(l);
}
