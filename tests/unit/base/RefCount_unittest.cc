/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Sanuj Sharma, Viktor Gal
 */

#include <shogun/lib/RefCount.h>
#if defined(HAVE_PTHREAD) || defined(HAVE_CXX11)

#ifdef HAVE_CXX11
#include <thread>
#elif HAVE_PTHREAD
#include <pthread.h>
#endif
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

#if !defined(HAVE_CXX11) && defined(HAVE_PTHREAD)
	pthread_exit(0);
#endif
	return NULL;
}

TEST(RefCount, stress_test)
{
	RefCount * rc = new RefCount(0);
	EXPECT_EQ(rc->ref_count(), 0);
	rc->ref();
	EXPECT_EQ(rc->ref_count(), 1);

#ifdef HAVE_CXX11
	std::thread threads[5];
#elif HAVE_PTHREAD
	pthread_t * threads = new pthread_t[5];
#endif
	for (index_t i = 0; i < 5; i++)
	{
#ifdef HAVE_CXX11
		threads[i] = std::thread(&stress_test_helper, static_cast<void *>(rc));
#elif HAVE_PTHREAD
		pthread_create(&threads[i], NULL, stress_test_helper, static_cast<void *>(rc));
#endif
	}

	for (index_t i = 0; i < 5; i++)
	{
#ifdef HAVE_CXX11
		threads[i].join();
#elif HAVE_PTHREAD
		pthread_join(threads[i], NULL);
#endif
	}

	EXPECT_EQ(rc->ref_count(), 1);
	rc->unref();
	EXPECT_EQ(rc->ref_count(), 0);
#if !defined(HAVE_CXX11) && defined(HAVE_PTHREAD)
	delete [] threads;
#endif
	delete rc;
}

#endif //  defined(HAVE_PTHREAD) || defined(HAVE_CXX11)
