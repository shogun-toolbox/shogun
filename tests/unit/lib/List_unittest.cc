/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Heiko Strathmann
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

#include <shogun/lib/List.h>
#include <gtest/gtest.h>

using namespace shogun;

#ifdef USE_REFERENCE_COUNTING
TEST(ListTest, contructor_ref_count_append_delete_data_true)
{
	// test reference counting of list
	CList* list=new CList(true);
	SG_REF(list);

	// element to add to list
	CList* data=new CList();
	SG_REF(data);
	ASSERT_TRUE(data->ref_count()==1);

	// add to list
	list->append_element(data);
	EXPECT_TRUE(data!=NULL);
	EXPECT_TRUE(data->ref_count()==2);

	// clean up
	SG_UNREF(list);
	SG_UNREF(data);
	ASSERT_TRUE(list==NULL);
	ASSERT_TRUE(data==NULL);
}

TEST(ListTest, contructor_ref_count_append_delete_data_false)
{
	// test reference counting of list
	CList* list=new CList(false);
	SG_REF(list);

	// element to add to list
	CList* data=new CList();
	SG_REF(data);
	ASSERT_TRUE(data->ref_count()==1);

	// add to list
	list->append_element(data);
	EXPECT_TRUE(data->ref_count()==1);

	// clean up
	SG_UNREF(list);
	SG_UNREF(data);

	ASSERT_TRUE(list==NULL);
	ASSERT_TRUE(data==NULL);
}

TEST(ListTest, destructur_ref_count_append_delete_data_true)
{
	// test reference counting of list
	CList* list=new CList(true);
	SG_REF(list);

	// element to add to list
	CList* data=new CList();
	SG_REF(data);
	ASSERT_TRUE(data->ref_count()==1);

	// add to list
	list->append_element(data);
	ASSERT_TRUE(data!=NULL);
	ASSERT_TRUE(data->ref_count()==2);

	// clean up
	SG_UNREF(list);
	EXPECT_TRUE(data!=NULL);
	EXPECT_TRUE(data->ref_count()==1);
	SG_UNREF(data);

	ASSERT_TRUE(list==NULL);
	ASSERT_TRUE(data==NULL);
}

TEST(ListTest, destructur_ref_count_append_delete_data_false)
{
	// test reference counting of list
	CList* list=new CList(false);
	SG_REF(list);

	// element to add to list
	CList* data=new CList();
	SG_REF(data);
	ASSERT_TRUE(data->ref_count()==1);

	// add to list
	list->append_element(data);
	ASSERT_TRUE(data!=NULL);
	ASSERT_TRUE(data->ref_count()==1);

	// clean up
	SG_UNREF(list);
	EXPECT_TRUE(data!=NULL);
	EXPECT_TRUE(data->ref_count()==1);
	SG_UNREF(data);

	ASSERT_TRUE(list==NULL);
	ASSERT_TRUE(data==NULL);
}

TEST(ListTest, get_first_element_identity)
{
	// test reference counting of list
	CList* list=new CList();
	SG_REF(list);

	// element to add to list
	CList* data=new CList();
	SG_REF(data);

	// add to list
	list->append_element(data);

	// get from list
	CSGObject* from_list=list->get_first_element();
	EXPECT_TRUE(from_list==data);

	// clean up
	SG_UNREF(list);
	SG_UNREF(data);

	ASSERT_TRUE(list==NULL);
	ASSERT_TRUE(data==NULL);
	ASSERT_TRUE(from_list!=NULL); // dead pointer
}

TEST(ListTest, get_first_element_ref_count_delete_data_true)
{
	// test reference counting of list
	CList* list=new CList(true);
	SG_REF(list);

	// element to add to list
	CList* data=new CList();
	SG_REF(data);
	ASSERT_TRUE(data->ref_count()==1);

	// add to list
	list->append_element(data);
	ASSERT_TRUE(data->ref_count()==2);

	// get from list
	CSGObject* from_list=list->get_first_element();
	ASSERT_TRUE(from_list==data);
	EXPECT_TRUE(from_list->ref_count()==3);

	// clean up
	SG_UNREF(list);
	SG_UNREF(data);
	SG_UNREF(from_list);

	ASSERT_TRUE(list==NULL);
	ASSERT_TRUE(data!=NULL); // dead pointer
	ASSERT_TRUE(from_list==NULL);
}

TEST(ListTest, get_first_element_ref_count_delete_data_false)
{
	// test reference counting of list
	CList* list=new CList(false);
	SG_REF(list);

	// element to add to list
	CList* data=new CList();
	SG_REF(data);
	ASSERT_TRUE(data->ref_count()==1);

	// add to list
	list->append_element(data);
	ASSERT_TRUE(data->ref_count()==1);

	// get from list
	CSGObject* from_list=list->get_first_element();
	ASSERT_TRUE(from_list==data);
	EXPECT_TRUE(from_list->ref_count()==1);
	SG_REF(from_list);

	// clean up
	SG_UNREF(list);
	SG_UNREF(data);
	SG_UNREF(from_list);

	ASSERT_TRUE(list==NULL);
	ASSERT_TRUE(data!=NULL); // dead pointer
	ASSERT_TRUE(from_list==NULL);
}
#endif //USE_REFERENCE_COUNTING
