/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Thoralf Klein
 */

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(CMath, qsort_test)
{
	// testing qsort on list of zero elements
	CMath::qsort((int32_t *)NULL, 0);

	// testing qsort on list of one element
	int32_t * v1 = SG_CALLOC(int32_t, 1);
	CMath::qsort(v1, 1);
	SG_FREE(v1);
}

TEST(CMath, qsort_ptr_test)
{
	// testing qsort on list of zero pointers
	CMath::qsort((int32_t **)NULL, 0);

	// testing qsort on list of one pointer
	int32_t ** v1 = SG_CALLOC(int32_t *, 1);
	CMath::qsort(v1, 1);
	SG_FREE(v1);
}

TEST(CMath, qsort_index_test)
{
	// testing qsort_index on list of zero elements
	CMath::qsort_index((int32_t *)NULL, (int32_t *)NULL, 0);

	// testing qsort_index on list of one element
	int32_t * v1 = SG_CALLOC(int32_t, 1);
	int32_t * i1 = SG_CALLOC(int32_t, 1);
	CMath::qsort_index(v1, i1, 1);
	SG_FREE(v1);
	SG_FREE(i1);
}

TEST(CMath, qsort_backward_index_test)
{
	// testing qsort_backward_index on list of zero elements
	CMath::qsort_backward_index((int32_t *)NULL, (int32_t *)NULL, 0);

	// testing qsort_backward_index on list of one element
	int32_t * v1 = SG_CALLOC(int32_t, 1);
	int32_t * i1 = SG_CALLOC(int32_t, 1);
	CMath::qsort_backward_index(v1, i1, 1);
	SG_FREE(v1);
	SG_FREE(i1);
}

TEST(CMath, parallel_qsort_index_test)
{
	// testing parallel_qsort_index on list of zero elements
	CMath::parallel_qsort_index((int32_t *)NULL, (int32_t *)NULL, 0, 8);

	// testing parallel_qsort_index on list of one element
	int32_t * v1 = SG_CALLOC(int32_t, 1);
	int32_t * i1 = SG_CALLOC(int32_t, 1);
	CMath::parallel_qsort_index(v1, i1, 1, 8);
	SG_FREE(v1);
	SG_FREE(i1);
}
