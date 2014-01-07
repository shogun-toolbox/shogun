#include <lib/RefCount.h>

using namespace shogun;

int32_t RefCount::ref()
{
#ifdef HAVE_CXX11_ATOMIC
	int32_t count = rc.fetch_add(1)+1;
#else
	lock.lock();
	int32_t count = ++rc;
	lock.unlock();
#endif

	return count;
}

int32_t RefCount::unref()
{
#ifdef HAVE_CXX11_ATOMIC
	int32_t count = rc.fetch_sub(1)-1;
#else
	lock.lock();
	int32_t count = --rc;
	lock.unlock();
#endif

	return count;
}

int32_t RefCount::ref_count()
{
#ifdef HAVE_CXX11_ATOMIC
	int32_t count = rc.load();
#else
	lock.lock();
	int32_t count = rc;
	lock.unlock();
#endif

	return count;
}
