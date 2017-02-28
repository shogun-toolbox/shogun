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
	int32_t ref();

	/** Decrease reference count
	 *
	 * @return the new reference count
	 */
	int32_t unref();

	/** Get the reference count
	 *
	 * @return the reference count
	 */
	int32_t ref_count();

	/** reference count */
    std::atomic<int32_t> rc;
};
}

#endif //_REFCOUNT__H__
