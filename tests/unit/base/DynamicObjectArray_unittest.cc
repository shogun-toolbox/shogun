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
	CDynamicObjectArray * orig_array = new CDynamicObjectArray();

	// Something relatively simple to add
	CSubset * subset = new CSubset();
	orig_array->append_element(subset);

	CDynamicObjectArray * cloned_array = (CDynamicObjectArray*) orig_array->clone();
	// Expand the cloned array into reserved space to check if the cloned
	// array has correctly allocated memory
	for (index_t i=0; i < 100; ++i)
		cloned_array->append_element(new CSubset());

	// Check sizes
	EXPECT_EQ(orig_array->get_num_elements(), 1);
	EXPECT_EQ(cloned_array->get_num_elements(), 101);

	SG_UNREF(orig_array);
	SG_UNREF(cloned_array);
}
