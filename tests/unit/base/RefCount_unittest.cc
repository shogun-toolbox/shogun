/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Sanuj Sharma, Viktor Gal
 */
#include <gtest/gtest.h>

#include <thread>
#include <shogun/lib/RefCount.h>

using namespace shogun;

void stress_test_helper(RefCount* rc)
{
	for (index_t i = 0; i < 10; i++)
	{
		rc->ref();
		rc->ref();
		rc->unref();
		rc->unref();
	}
}

TEST(RefCount, stress_test)
{
	auto rc = new RefCount(0);
	EXPECT_EQ(rc->ref_count(), 0);
	rc->ref();
	EXPECT_EQ(rc->ref_count(), 1);

	std::thread threads[5];
	for (index_t i = 0; i < 5; i++)
		threads[i] = std::thread(stress_test_helper, rc);

	for (index_t i = 0; i < 5; i++)
		threads[i].join();

	EXPECT_EQ(rc->ref_count(), 1);
	rc->unref();
	EXPECT_EQ(rc->ref_count(), 0);
	delete rc;
}
