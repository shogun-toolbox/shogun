/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Thoralf Klein
 */

#include <shogun/lib/RefCount.h>
#include <pthread.h>
#include <gtest/gtest.h>

using namespace shogun;

void * stress_test_helper(void * args)
{
	RefCount * rc = (RefCount *) args;

	for (index_t i = 0; i < 10; i++)
	{
		rc->ref();
		rc->ref();
		rc->unref();
		rc->unref();
	}

	pthread_exit(0);
}

TEST(RefCount, stress_test)
{
	RefCount * rc = new RefCount(0);
	EXPECT_EQ(rc->ref_count(), 0);
	rc->ref();
	EXPECT_EQ(rc->ref_count(), 1);

	pthread_t * threads = new pthread_t[5];

	for (index_t i = 0; i < 5; i++)
	{
		pthread_create(&threads[i], NULL, stress_test_helper, static_cast<void *>(rc));
	}

	for (index_t i = 0; i < 5; i++)
	{
		pthread_join(threads[i], NULL);
	}

	EXPECT_EQ(rc->ref_count(), 1);
	rc->unref();
	EXPECT_EQ(rc->ref_count(), 0);
	delete [] threads;
	delete rc;
}
