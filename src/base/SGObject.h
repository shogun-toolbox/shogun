/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max Planck Society
 */

#ifndef __SGOBJECT_H__
#define __SGOBJECT_H__

#include "lib/io.h"
#include "base/Parallel.h"
#include "base/Version.h"

class CSGObject;
class CIO;

// define reference counter macros
#define SG_REF(x) { if (x) x->ref(); }
#define SG_UNREF(x) { if (x) x->unref(); } 

/** Class SGObject is the base class of all shogun objects. Apart from dealing
 * with reference counting that is used to manage shogung objects in memory
 * (erase unused object, avoid cleaning objects when they are still in use), it
 * provides interfaces for:
 * -# parallel - to determine the number of used CPUs for a method (cf. CParallel)
 * -# io - to output messages and general i/o (cf. CIO)
 * -# version - to provide version information of the shogun version used (cf. CVersion)
 */
class CSGObject
{
public:
	inline CSGObject() : refcount(0)
	{
		set_global_objects();
	}

	inline CSGObject(const CSGObject& orig) : refcount(0), io(orig.io),
		parallel(orig.parallel), version(orig.version)
	{
		set_global_objects();
	}

    virtual ~CSGObject()
	{
		SG_UNREF(parallel);
		SG_UNREF(io);
		SG_UNREF(version);
	}

	inline int32_t ref()
	{
		++refcount;
		SG_DEBUG("ref():%ld obj:%p\n", refcount, this);
		return refcount;
	}

	inline int32_t ref_count() const
	{
		SG_DEBUG("ref_count(): refcount is: %d\n", refcount);
		return refcount;
	}

	inline int32_t unref()
	{
		if (refcount==0 || --refcount==0)
		{
			SG_DEBUG("unref():%ld obj:%p destroying\n", refcount, this);
			//don't do this yet for static interfaces (as none is
			//calling ref/unref properly)
			delete this;
			return 0;
		}
		else
		{
			SG_DEBUG("unref():%ld obj:%p decreased\n", refcount, this);
			return refcount;
		}
	}

private:
	void set_global_objects();

private:
	int32_t refcount;
	bool static_io;

public:
	CIO* io;
	CParallel* parallel;
	CVersion* version;
};
#endif // __SGOBJECT_H__
