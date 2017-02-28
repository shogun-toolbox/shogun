#include <shogun/lib/RefCount.h>

using namespace shogun;

int32_t RefCount::ref()
{
	return rc.fetch_add(1, std::memory_order_relaxed)+1;
}

int32_t RefCount::unref()
{
	return rc.fetch_sub(1, std::memory_order_acquire)-1;
}

int32_t RefCount::ref_count()
{
	return rc.load(std::memory_order_acquire);
}
