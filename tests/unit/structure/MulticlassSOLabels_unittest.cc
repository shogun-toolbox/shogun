/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2013 Zuse-Institute-Berlin (ZIB)
 * Copyright (C) 2013 Thoralf Klein
 */

#include <shogun/base/init.h>
#include <shogun/structure/MulticlassSOLabels.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(MulticlassSOLabels, create)
{
	RealNumber* real_number = new RealNumber(13);
	SG_REF(real_number);

	EXPECT_EQ(13, real_number->value);

	SG_UNREF(real_number);
}

TEST(MulticlassSOLabels, obtain_from_generic)
{
	RealNumber* real_number = new RealNumber(13);
	SG_REF(real_number);

	StructuredData* generic_label = (StructuredData *) real_number;
	RealNumber* real_number_casted = RealNumber::obtain_from_generic(generic_label);
	EXPECT_EQ(13, real_number_casted->value);

	SG_UNREF(real_number);
}
