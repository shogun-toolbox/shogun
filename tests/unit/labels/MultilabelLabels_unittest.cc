/*
 * Copyright (C) 2013 Zuse-Institute-Berlin (ZIB)
 * Copyright (C) 2013-2014 Thoralf Klein
 * Written (W) 2013-2014 Thoralf Klein
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/base/init.h>
#include <shogun/labels/MultilabelLabels.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(MultilabelLabels, constructor_0)
{
	CMultilabelLabels * ml = new CMultilabelLabels();
	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 0);
	ml->ensure_valid("unittest");

	SG_UNREF(ml);
}


TEST(MultilabelLabels, constructor_1)
{
	CMultilabelLabels * ml = new CMultilabelLabels(6);
	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 0);
	EXPECT_EQ(ml->get_num_classes(), 6);
	ml->ensure_valid("unittest");

	SG_UNREF(ml);
}


TEST(MultilabelLabels, constructor_2)
{
	CMultilabelLabels * ml = new CMultilabelLabels(5,6);
	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 5);
	EXPECT_EQ(ml->get_num_classes(), 6);
	ml->ensure_valid("unittest");

	SG_UNREF(ml);
}


TEST(MultilabelLabels, clone)
{
	CMultilabelLabels * ml = new CMultilabelLabels(5,6);
	ASSERT_TRUE(ml != NULL);

	CMultilabelLabels * mlc = (CMultilabelLabels *)ml->clone();
	ASSERT_TRUE(mlc != NULL);

	EXPECT_EQ(ml->get_num_labels(), 5);
	EXPECT_EQ(ml->get_num_classes(), 6);

	SG_UNREF(ml);
	SG_UNREF(mlc);
}


TEST(MultilabelLabels, to_dense_2)
{
	SGVector<int16_t> sparse(2);
	sparse[0] = 2;
	sparse[1] = 5;

	EXPECT_EQ(2, sparse.size());

	SGVector<float64_t> dense = CMultilabelLabels::to_dense<int16_t, float64_t> (&sparse, 20, +1, 0);
	EXPECT_EQ(20, dense.size());
	EXPECT_EQ(+1, dense[2]);
	EXPECT_EQ(+1, dense[5]);
	EXPECT_EQ(+2, SGVector<float64_t>::sum(dense.vector, dense.vlen));
}


TEST(MultilabelLabels, get_label)
{
	CMultilabelLabels * ml = new CMultilabelLabels(10,5);
	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 10);
	EXPECT_EQ(ml->get_num_classes(), 5);

	for (int32_t i=0; i<ml->get_num_labels(); i++)
	{
		SGVector<int16_t> sparse = ml->get_label(i);
		EXPECT_EQ(0, sparse.size());

		SGVector<float64_t> dense = CMultilabelLabels::to_dense<int16_t, float64_t> (&sparse, ml->get_num_labels(), +1, -1);
		EXPECT_EQ(ml->get_num_labels(), dense.size());
	}

	// TODO: Check for failure:
	// CBinaryLabels label_invalid = ml->get_binary_for_label(5);

	SG_UNREF(ml);
}


TEST(MultilabelLabels, get_class_labels)
{
	CMultilabelLabels * ml = new CMultilabelLabels(10,5);
	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 10);
	EXPECT_EQ(ml->get_num_classes(), 5);

	SGVector<int32_t> ** class_labels = ml->get_class_labels();
	ASSERT_TRUE(class_labels != NULL);

	for (int32_t i=0; i<ml->get_num_classes(); i++)
	{
		EXPECT_EQ(0, class_labels[i]->size());

		SGVector<float64_t> dense = CMultilabelLabels::to_dense<int32_t, float64_t> (class_labels[i], ml->get_num_labels(), +1, -1);

		EXPECT_EQ(ml->get_num_labels(), dense.size());
		delete class_labels[i];
	}

	SG_FREE(class_labels);
	SG_UNREF(ml);
}


TEST(MultilabelLabels, set_class_labels)
{
	const int32_t num_labels = 5;
	const int32_t num_classes = 7;

	SGVector<int32_t> ** class_labels = SG_MALLOC(SGVector<int32_t> *, num_classes);
	for (int32_t i=0; i<num_classes; i++)
	{
		class_labels[i] = new SGVector<int32_t>(3);
		class_labels[i]->set_const(0);

		(*class_labels[i])[0] = 0;
		(*class_labels[i])[1] = i%num_labels;
		(*class_labels[i])[2] = (i*i)%num_labels;

		int32_t new_size = SGVector<int32_t>::unique(class_labels[i]->vector, class_labels[i]->vlen);
		class_labels[i]->vlen = new_size;
		// class_labels[i]->display_vector("yC");
	}

	CMultilabelLabels * ml = new CMultilabelLabels(num_labels, num_classes);
	ASSERT_TRUE(ml != NULL);

	ml->set_class_labels(class_labels);
	ASSERT_EQ(num_labels, ml->get_num_labels());
	ASSERT_EQ(num_classes, ml->get_num_classes());

	for (int32_t i=0; i<num_classes; i++)
	{
		delete class_labels[i];
	}
	SG_FREE(class_labels);

	// ml->display();

	ASSERT_EQ(7, ml->get_label(0).size());
	EXPECT_EQ(0, (ml->get_label(0))[0]);
	EXPECT_EQ(1, (ml->get_label(0))[1]);
	EXPECT_EQ(2, (ml->get_label(0))[2]);
	EXPECT_EQ(6, (ml->get_label(0))[6]);

	ASSERT_EQ(3, ml->get_label(1).size());
	EXPECT_EQ(1, (ml->get_label(1))[0]);
	EXPECT_EQ(4, (ml->get_label(1))[1]);
	EXPECT_EQ(6, (ml->get_label(1))[2]);

	ASSERT_EQ(1, ml->get_label(2).size());
	EXPECT_EQ(2, (ml->get_label(2))[0]);

	ASSERT_EQ(1, ml->get_label(3).size());
	EXPECT_EQ(3, (ml->get_label(3))[0]);

	ASSERT_EQ(3, ml->get_label(4).size());
	EXPECT_EQ(2, (ml->get_label(4))[0]);
	EXPECT_EQ(3, (ml->get_label(4))[1]);
	EXPECT_EQ(4, (ml->get_label(4))[2]);

	SG_UNREF(ml);
}


TEST(MultilabelLabels, set_class_labels_overflow)
{
	const int32_t num_labels = 65538;
	const int32_t num_classes = 3;

	SGVector<int32_t> ** class_labels = SG_MALLOC(SGVector<int32_t> *, num_classes);

	class_labels[0] = new SGVector<int32_t>(1);
	class_labels[0]->set_const(0);
	// class_labels[0]->display_vector("v_0");

	class_labels[1] = new SGVector<int32_t>(1);
	class_labels[1]->set_const(0);
	// class_labels[1]->display_vector("v_1");

	class_labels[2] = new SGVector<int32_t>(32768);
	class_labels[2]->set_const(0);

	SGVector<int32_t>::range_fill_vector(class_labels[2]->vector, class_labels[2]->vlen, 1);
	// class_labels[2]->display_vector("v_2");

	CMultilabelLabels * ml = new CMultilabelLabels(num_labels, num_classes);
	ASSERT_TRUE(ml != NULL);

	ml->set_class_labels(class_labels);
	// ml->display();

	ASSERT_EQ(num_labels, ml->get_num_labels());
	ASSERT_EQ(num_classes, ml->get_num_classes());

	ASSERT_EQ(2, ml->get_label(0).size());
	EXPECT_EQ(0, (ml->get_label(0))[0]);
	EXPECT_EQ(1, (ml->get_label(0))[1]);

	ASSERT_EQ(1, ml->get_label(1).size());
	EXPECT_EQ(2, (ml->get_label(1))[0]);

	ASSERT_EQ(1, ml->get_label(32768).size());
	EXPECT_EQ(2, (ml->get_label(32768))[0]);

	ASSERT_EQ(0, ml->get_label(32769).size());
	ASSERT_EQ(0, ml->get_label(num_labels-1).size());

	for (int32_t i=0; i<num_classes; i++)
	{
		delete class_labels[i];
	}
	SG_FREE(class_labels);

	SG_UNREF(ml);
}


TEST(MultilabelLabels, display)
{
	CMultilabelLabels * ml = new CMultilabelLabels(10,5);
	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 10);
	EXPECT_EQ(ml->get_num_classes(), 5);

	ml->display();

	SG_UNREF(ml);
}
