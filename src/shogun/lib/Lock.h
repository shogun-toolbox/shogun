/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2013 Soeren Sonnenburg
 */
#ifndef __LOCK_H__
#define __LOCK_H__

#include <shogun/lib/config.h>

namespace shogun
{
/** @brief Class Lock used for synchronization in concurrent programs. */
class CLock
{
public:
	/** default constructor */
	CLock();
	/** de-structor */
	~CLock();

	/** lock the object */
	void lock();
	/** unlock the object (must be called as often as lock) */
	void unlock();

private:
	/** lock object */
	void* lock_object;
};
}
#endif // __LOCK_H__
