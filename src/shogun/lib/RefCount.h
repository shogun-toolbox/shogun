#ifndef _REFCOUNT__H__
#define _REFCOUNT__H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <atomic>

namespace shogun
{
/** brief This class implements a thread-safe counter used for
 * reference counting.
 */
class RefCount
{
public:
	/** Constructor
	 *
	 * @param ref_start starting value for counter
	 */
	RefCount(int32_t ref_start=0) : rc(ref_start) {};

	/** Increase ref count
	 *
	 * @return the new reference count
	 */
	SG_FORCED_INLINE int32_t ref()
	{
		return rc.fetch_add(1, std::memory_order_relaxed)+1;
	}

	/** Decrease reference count
	 *
	 * @return the new reference count
	 */
	SG_FORCED_INLINE int32_t unref()
	{
		return rc.fetch_sub(1, std::memory_order_acquire)-1;
	}

	/** Get the reference count
	 *
	 * @return the reference count
	 */
	SG_FORCED_INLINE int32_t ref_count()
	{
		return rc.load(std::memory_order_acquire);
	}

private:
	/** reference count */
    std::atomic<int32_t> rc;
};
}

#endif //_REFCOUNT__H__
