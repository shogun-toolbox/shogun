/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Thoralf Klein <thoralf.klein@zib.de>
 * Copyright (C) 2013 Zuse-Institute-Berlin (ZIB)
 * Copyright (C) 2013 Thoralf Klein
 */

#include <shogun/base/init.h>
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

  EXPECT_EQ(3, l->get_num_labels());
  EXPECT_EQ(3, CRealNumber::obtain_from_generic(l->get_label(0))->value);
  EXPECT_EQ(7, CRealNumber::obtain_from_generic(l->get_label(1))->value);
  EXPECT_EQ(13, CRealNumber::obtain_from_generic(l->get_label(2))->value);

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

  EXPECT_EQ(3, l->get_num_labels());
  EXPECT_EQ(3, CRealNumber::obtain_from_generic(l->get_label(0))->value);
  EXPECT_EQ(23, CRealNumber::obtain_from_generic(l->get_label(1))->value);
  EXPECT_EQ(13, CRealNumber::obtain_from_generic(l->get_label(2))->value);

  SG_UNREF(l);
}
