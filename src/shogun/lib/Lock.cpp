/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2013 Soeren Sonnenburg
 */
#include <shogun/lib/config.h>
#include <shogun/lib/memory.h>
#include <shogun/lib/Lock.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#ifdef USE_SPINLOCKS
#ifdef DARWIN
#include <libkern/OSAtomic.h>
	#define PTHREAD_LOCK_T OSSpinLock
	#define PTHREAD_LOCK_INIT(lock) *lock = OS_SPINLOCK_INIT
	#define PTHREAD_LOCK_DESTROY(lock)
	#define PTHREAD_LOCK(lock) OSSpinLockLock(lock)
	#define PTHREAD_UNLOCK(lock) OSSpinLockUnlock(lock)
#else
	#define PTHREAD_LOCK_T pthread_spinlock_t
	#define PTHREAD_LOCK_INIT(lock) pthread_spin_init(lock, 0)
	#define PTHREAD_LOCK_DESTROY(lock) pthread_spin_destroy(lock)
	#define PTHREAD_LOCK(lock) pthread_spin_lock(lock)
	#define PTHREAD_UNLOCK(lock) pthread_spin_unlock(lock)
#endif // DARWIN
#else
	#define PTHREAD_LOCK_T pthread_mutex_t
	#define PTHREAD_LOCK_INIT(lock) pthread_mutex_init(lock, NULL)
	#define PTHREAD_LOCK_DESTROY(lock) pthread_mutex_destroy(lock)
	#define PTHREAD_LOCK(lock) pthread_mutex_lock(lock)
	#define PTHREAD_UNLOCK(lock) pthread_mutex_unlock(lock)
#endif // USE_SPINLOCKS
#endif // HAVE_PTHREAD

using namespace shogun;

CLock::CLock()
{
#if !defined(HAVE_CXX11_ATOMIC) && defined(HAVE_PTHREAD)
	lock_object=(void*) SG_MALLOC(PTHREAD_LOCK_T, 1);
	PTHREAD_LOCK_INIT((PTHREAD_LOCK_T*) lock_object);
#endif
}

CLock::~CLock()
{
#if !defined(HAVE_CXX11_ATOMIC) && defined(HAVE_PTHREAD)
	PTHREAD_LOCK_DESTROY((PTHREAD_LOCK_T*) lock_object);
	SG_FREE(lock_object);
#endif
}

void CLock::lock()
{
#ifdef HAVE_CXX11_ATOMIC
	while(m_flag.test_and_set(std::memory_order_acquire));
#elif HAVE_PTHREAD
	PTHREAD_LOCK((PTHREAD_LOCK_T*) lock_object);
#else
	SG_NOTIMPLEMENTED
#endif
}

void CLock::unlock()
{
#ifdef HAVE_CXX11_ATOMIC
	m_flag.clear(std::memory_order_release);
#elif HAVE_PTHREAD
	PTHREAD_UNLOCK((PTHREAD_LOCK_T*) lock_object);
#else
	SG_NOTIMPLEMENTED
#endif
}
