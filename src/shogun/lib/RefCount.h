#ifdef HAVE_CXX11_ATOMIC
#include <atomic>
#endif

#include <shogun/lib/common.h>
#include <shogun/lib/Lock.h>

#ifndef _REFCOUNT__H__
#define _REFCOUNT__H__

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
	RefCount(int32_t ref_start=0) : rc(ref_start) {}

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
#ifdef HAVE_CXX11_ATOMIC
    volatile std::atomic<int> rc;
#else
	int32_t rc;

	/** the lock */
	CLock lock;
#endif
};
}

#endif //_REFCOUNT__H__
