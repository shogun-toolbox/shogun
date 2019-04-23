/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2017 Leon Kuchenbecker
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

#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/features/Subset.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(DynamicObjectArray,clone)
{
	auto orig_array = std::make_shared<DynamicObjectArray>();

	// Something relatively simple to add
	auto subset = std::make_shared<Subset>();
	orig_array->append_element(subset);

	auto cloned_array = orig_array->clone()->as<DynamicObjectArray>();
	// Expand the cloned array into reserved space to check if the cloned
	// array has correctly allocated memory
	for (index_t i=0; i < 100; ++i)
		cloned_array->append_element(std::make_shared<Subset>());

	// Check sizes
	EXPECT_EQ(orig_array->get_num_elements(), 1);
	EXPECT_EQ(cloned_array->get_num_elements(), 101);



}

TEST(DynamicObjectArray, equals_after_resize)
{
	auto array1 = std::make_shared<DynamicObjectArray>();
	auto array2 = std::make_shared<DynamicObjectArray>();

	/* enforce a resize */
	for (index_t i = 0; i < 1000; ++i)
		array1->append_element(std::make_shared<DynamicObjectArray>());

	array1->reset_array();

	EXPECT_TRUE(array1->equals(array2));
	EXPECT_TRUE(array2->equals(array1));



}

TEST(DynamicObjectArray, equals_different)
{
	auto array1 = std::make_shared<DynamicObjectArray>();
	auto array2 = std::make_shared<DynamicObjectArray>();

	array1->append_element(std::make_shared<DynamicObjectArray>());

	EXPECT_FALSE(array1->equals(array2));
	EXPECT_FALSE(array2->equals(array1));



}
