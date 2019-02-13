/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Soeren Sonnenburg, Fernando Iglesias, Yuyu Zhang
 */
#ifndef __LOCK_H__
#define __LOCK_H__

#include <shogun/lib/config.h>
#include <shogun/lib/cpu.h>
#include <atomic>

namespace shogun
{
/** @brief Class Lock used for synchronization in concurrent programs. */
class SHOGUN_EXPORT CLock
{
public:
	/** lock the object */
	SG_FORCED_INLINE void lock()
	{
		do
		{
			while (m_locked.load(std::memory_order_relaxed))
				CpuRelax();
		}
		while (m_locked.exchange(true, std::memory_order_acquire));
	}

	/** unlock the object (must be called as often as lock) */
	SG_FORCED_INLINE void unlock()
	{
		m_locked.store(false, std::memory_order_release);
	}

private:
	/** lock object */
	std::atomic_bool m_locked = { false };
};
}
#endif // __LOCK_H__
