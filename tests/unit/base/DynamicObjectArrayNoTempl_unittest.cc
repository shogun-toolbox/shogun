/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Written (W) 2017 Leon Kuchenbecker
 */

#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/features/Subset.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(DynamicObjectArray,clone)
{
	// Base array
	CDynamicObjectArray * do_array = new CDynamicObjectArray();

	// Something relatively simple to add
	CSubset * subset = new CSubset();
	do_array->append_element(subset);

	// Cloned array
	CDynamicObjectArray * do_array_clone = (CDynamicObjectArray*) do_array->clone();

	// Expand the cloned array
	for (size_t i=0; i < 100; ++i)
		do_array_clone->append_element(new CSubset());

	// Check sizes
	EXPECT_EQ(do_array->get_num_elements(), 1);
	EXPECT_EQ(do_array_clone->get_num_elements(), 101);

	SG_UNREF(do_array);
	SG_UNREF(do_array_clone);
}
