/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <base/init.h>
#include <features/SubsetStack.h>

using namespace shogun;

void test()
{
	CSubsetStack* stack=new CSubsetStack();
	SG_REF(stack);

	/* subset indices, each set is shifted by one */
	SGVector<index_t> subset_a(10);
	SGVector<index_t> subset_b(4);
	subset_a.range_fill(1);
	subset_b.range_fill(1);

	/* add and remove subsets a couple of times */
	stack->add_subset(subset_a);
	stack->remove_subset();
	stack->add_subset(subset_b);
	stack->remove_subset();

	/* add and remove subsets a couple of times, different order */
	stack->add_subset(subset_a);
	stack->add_subset(subset_b);
	stack->remove_subset();
	stack->remove_subset();

	/** add two subsets and check if index mapping works */
	stack->add_subset(subset_a);
	stack->add_subset(subset_b);

	/* remember, offset of one for each index set */
	for (index_t i=0; i<subset_b.vlen; ++i)
		ASSERT(stack->subset_idx_conversion(i)==i+2);

	stack->remove_subset();
	stack->remove_subset();

	/* clean up */
	SG_UNREF(stack);
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	test();

	exit_shogun();

	return 0;
}

